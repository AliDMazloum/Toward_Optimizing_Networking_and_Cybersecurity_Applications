#!/usr/bin/env bash
#
# Measure what the current App2 program costs against the original one, on this
# machine, at one configuration.
#
# The original program is Old_files/App2/DPI_v7.2.cu. Its problem size is fixed
# at compile time, it uses the halfword intrinsic unconditionally, and it
# launches 64 threads per block over half its signature count. The current
# program takes its size at run time, so the comparison is only meaningful at
# the size the original was compiled for.
#
# That size is therefore read out of the original source rather than written
# here. Setting it in two places would let the two drift apart, and a control
# that compares one size against another reports a ratio that is mostly the
# size difference. Whatever the constants say is what both arms run.
#
# Both programs are run as separate processes, one timed launch each, because
# the original has no trial loop and its warm-up is commented out: it measures
# a cold launch. Comparing that against a warm mean would charge the difference
# to the program instead of to the clock state. The first run of each arm is
# discarded, the rest are averaged.
#
# Four arms, so that three separate effects can be told apart:
#
#   A  original, cold single launch
#   B  current, cold single launch, 64 threads per block, the same geometry
#   C  current, cold single launch, 32 threads per block, the swept geometry
#   D  current, the swept protocol, five trials after one warm-up
#
# B over A is what the current program costs at matched geometry. C over B is
# what the block size costs. D against C is what a warm clock is worth. The
# swept number is D.
#
# Usage, from the root of the clone:
#
#   ./debug/run_overhead_control.sh a100
#   ./debug/run_overhead_control.sh h200
#
# The original program writes a signatures.txt of about 160 MB into the working
# directory on every run and does not remove it. This script runs each arm in
# its own scratch directory and deletes it at the end.

set -u

MACHINE=${1:-}
REPS=${REPS:-6}

case "$MACHINE" in
    a100) ARCHES=${ARCHES:-80}; TAG=a100; GPU_MATCH=A100 ;;
    h200) ARCHES=${ARCHES:-90}; TAG=h200; GPU_MATCH=H200 ;;
    *)
        echo "Usage: $0 <a100|h200>" >&2
        exit 2
        ;;
esac

NEW=App2/smith_waterman_dpi-$TAG
OLD_SRC=Old_files/App2/DPI_v7.2.cu
OLD=Old_files/App2/DPI_v7.2-$TAG
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/overhead.XXXXXX")
ROOT=$(pwd)

# The configuration, read from the original source so that the two arms cannot
# run different sizes.
constant() {   # constant <name>
    sed -n "s/^#define $1  *\([0-9][0-9]*\).*/\1/p" "$OLD_SRC" | head -1
}
SIGS=$(constant NumberOfSignatures)
PAY=$(constant PayloadSize)
LEN=$(constant MaxSignatureLength)
if [ -z "$SIGS" ] || [ -z "$PAY" ] || [ -z "$LEN" ]; then
    echo "Could not read the size constants out of $OLD_SRC. It defines" >&2
    echo "NumberOfSignatures, PayloadSize and MaxSignatureLength; one of them" >&2
    echo "is missing or is no longer a plain integer." >&2
    exit 1
fi

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$name" in
    *$GPU_MATCH*) echo "Node gpu: $name" ;;
    *)
        echo "This node reports '$name', not a $GPU_MATCH. Refusing to run." >&2
        exit 1
        ;;
esac
echo "Scratch: $SCRATCH, $REPS runs per arm, first of each discarded."
echo "Configuration, read from $OLD_SRC:"
echo "  $SIGS signatures, $PAY byte payload, $LEN byte signatures"
if [ "$SIGS" -lt 5000000 ]; then
    echo
    echo "  NOTE this is a short run. The H200 figure this control is compared"
    echo "       against was taken at 10,000,000 signatures, where the kernel"
    echo "       runs about a second on the A100 and cold-launch noise is a"
    echo "       small fraction of it. Below that the noise grows and the two"
    echo "       numbers are not comparable with the earlier one. Restore the"
    echo "       constants in the original source to compare against it."
fi
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "Build"
echo "=============================================================="
make ARCHES=$ARCHES TAG=$TAG app2 || exit 1
if [ ! -x "$NEW" ]; then
    echo "Build did not produce $NEW." >&2
    exit 1
fi

# The original is built here rather than through the Makefile, because it is
# not one of the programs the Makefile maintains. Same optimisation level and
# same architecture as the current program; it needs neither NVML nor OpenMP.
${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES \
    "$OLD_SRC" -o "$OLD" || exit 1

# The pinned build. The original's problem size is a compile-time constant, so
# its compiler knows the payload loop's trip count and the threshold; the
# current program takes both at run time and its compiler does not. The kernel
# carries PIN_ macros that put those back, and this build turns them on, so the
# difference between arm B and arm E is what the run-time parameters cost. The
# threshold is floor(alpha * signature length) with the swept alpha of 0.8, so
# it is derived here rather than typed.
THRESHOLD=$(awk -v l="$LEN" 'BEGIN { printf "%d", int(0.8 * l) }')
PINNED=App2/smith_waterman_dpi-$TAG-pinned
${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES \
    -DPIN_SIGNATURES=$SIGS -DPIN_PAYLOAD=$PAY -DPIN_THRESHOLD=$THRESHOLD \
    -DPIN_EXIT_FIRST=1 App2/smith_waterman_dpi.cu \
    ${NVML_LIBS:--lnvidia-ml -lpthread} -o "$PINNED" || exit 1

# The payload-only build. All four pins together fix a binary to one point of
# the sweep, which would mean sixteen binaries per card. Only one of them is a
# loop bound: the payload length is the outer loop's trip count, and the sweep
# uses exactly two payload lengths. If this build matches the fully pinned one,
# the whole cost is that single bound and two binaries per card cover
# everything, which is the difference between a small change and a large one.
PAYPIN=App2/smith_waterman_dpi-$TAG-paypin
${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES \
    -DPIN_PAYLOAD=$PAY App2/smith_waterman_dpi.cu \
    ${NVML_LIBS:--lnvidia-ml -lpthread} -o "$PAYPIN" || exit 1

echo "Built $NEW, $PINNED, $PAYPIN and $OLD"
echo

# Registers and spills, from the compiler rather than from a run. A kernel that
# spills to local memory is slower for a reason the timings alone cannot show.
# The current program instantiates its kernel sixteen times, so the mangled
# template arguments are translated: sw_scan<signature length, intrinsic on,
# rows in registers, regex mode>. Only one of the sixteen is measured here, the
# one this control's configuration selects.
regs() {   # regs <source> [extra flags]
    src=$1; shift
    ${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES \
        "$@" -Xptxas -v -c "$src" -o /dev/null 2>&1 \
        | awk '
            /Function properties for/ {
                label = $0
                sub(/.*Function properties for /, "", label)
                # sw_scan<signature length, intrinsic, rows in registers, regex>
                if (match(label, /_Z7sw_scanILi[0-9]+ELb[01]ELb[01]ELb[01]E/)) {
                    a = substr(label, RSTART, RLENGTH)
                    gsub(/[^0-9]/, " ", a)
                    split(a, f, " ")
                    label = sprintf("sw_scan<len %s, dpx %s, regrows %s, regex %s>",
                                    f[2], f[3], f[4], f[5])
                }
                spill = ""
                next
            }
            /spill stores/ {
                if ($0 !~ /0 bytes spill stores/) spill = "  SPILLS"
                next
            }
            /Used [0-9]+ registers/ {
                for (i = 1; i <= NF; i++) if ($i == "Used") u = $(i + 1)
                printf "    %-48s %3d registers%s\n", label, u, spill
            }'
}

echo "Register use for sm_$ARCHES, from ptxas:"
echo "  $OLD_SRC"
regs "$OLD_SRC"
echo "  App2/smith_waterman_dpi.cu, unpinned"
regs App2/smith_waterman_dpi.cu
echo "  App2/smith_waterman_dpi.cu, payload pinned to $PAY"
regs App2/smith_waterman_dpi.cu -DPIN_PAYLOAD=$PAY
echo

# ---------------------------------------------------------------------------
# mean_of <file>   reads one number per line, drops the first, prints the
#                  median, the mean, the smallest, the largest, the spread as a
#                  percentage, and the count.
#
# The median leads because this card's readings are not unimodal: a cold clock
# boosts for whole trials and then settles, so a run of five holds a cluster of
# repeated sustained values and one or two fast excursions. The mean of that is
# neither. The sustained figure is what gets reported, by the same rule the
# routing measurements follow, and with five readings the median is the closest
# robust estimate of it. Where the two agree the distribution is unimodal and
# the distinction does not arise.
mean_of() {
    awk 'NR > 1 { v[++n] = $1; s += $1 }
         END {
             if (n == 0) { print "no runs"; exit }
             m = s / n
             for (i = 2; i <= n; i++) {       # insertion sort, n is 5
                 x = v[i]
                 for (j = i - 1; j >= 1 && v[j] > x; j--) v[j + 1] = v[j]
                 v[j + 1] = x
             }
             med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
             printf "%.6f %.6f %.6f %.6f %.2f %d",
                    med, m, v[1], v[n], (v[n] - v[1]) / m * 100, n
         }' "$1"
}

report() {   # report <label> <file>
    set -- "$1" $(mean_of "$2")
    if [ "$2" = "no" ]; then
        printf "  %-38s did not run\n" "$1"
        return
    fi
    printf "  %-34s median %s  mean %s  range %s to %s  spread %s %%  n=%s\n" \
           "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "Arm A: the original program, cold single launch"
echo "=============================================================="
mkdir -p "$SCRATCH/a"
: > "$SCRATCH/A.times"
for i in $(seq 1 $REPS); do
    t=$(cd "$SCRATCH/a" && "$ROOT/$OLD" 2>&1 \
        | sed -n 's/^time taken by the GPU kernel is \([0-9.]*\) s$/\1/p')
    if [ -z "$t" ]; then
        echo "  run $i produced no time; the arm is abandoned." >&2
        break
    fi
    echo "$t" >> "$SCRATCH/A.times"
    printf "  run %d: %s s\n" "$i" "$t"
done
echo

# ---------------------------------------------------------------------------
# new_arm <label> <outfile> <block> <trials> <warmup> <binary>
new_arm() {
    label=$1; out=$2; block=$3; trials=$4; warmup=$5; bin=${6:-$NEW}
    : > "$out"
    for i in $(seq 1 $REPS); do
        # No --verify here, so every arm runs with the same default the sweep
        # ran with. Verification happens on the host after the kernel and is
        # outside the timed window in any case.
        t=$(./$bin --signatures $SIGS --payload $PAY --sig-len $LEN \
                --mode literal --rows registers --dpx on --block $block \
                --trials $trials --warmup $warmup 2>&1 \
            | sed -n 's/^# kernel_s  *mean \([0-9.]*\) .*/\1/p')
        if [ -z "$t" ]; then
            echo "  run $i produced no time; the arm is abandoned." >&2
            break
        fi
        echo "$t" >> "$out"
        printf "  run %d: %s s\n" "$i" "$t"
    done
}

echo "=============================================================="
echo "Arm B: the current program, cold single launch, 64 threads per block"
echo "=============================================================="
new_arm B "$SCRATCH/B.times" 64 1 0
echo

echo "=============================================================="
echo "Arm C: the current program, cold single launch, 32 threads per block"
echo "=============================================================="
new_arm C "$SCRATCH/C.times" 32 1 0
echo

echo "=============================================================="
echo "Arm D: the current program, the swept protocol"
echo "=============================================================="
new_arm D "$SCRATCH/D.times" 32 5 1
echo

echo "=============================================================="
echo "Arm E: the current program pinned, cold single launch, block 64"
echo "=============================================================="
new_arm E "$SCRATCH/E.times" 64 1 0 "$PINNED"
echo

echo "=============================================================="
echo "Arm F: the current program with only the payload pinned, block 64"
echo "=============================================================="
new_arm F "$SCRATCH/F.times" 64 1 0 "$PAYPIN"
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "What happened"
echo "=============================================================="
echo "  Every ratio below is taken on the medians, for the reason given above"
echo "  the statistic."
echo
report "A original, cold, block 64" "$SCRATCH/A.times"
report "B current, cold, block 64" "$SCRATCH/B.times"
report "C current, cold, block 32" "$SCRATCH/C.times"
report "D current, swept protocol" "$SCRATCH/D.times"
report "E current all pins, cold, block 64" "$SCRATCH/E.times"
report "F current payload pin, cold, block 64" "$SCRATCH/F.times"
echo

ratio() {   # ratio <label> <numerator file> <denominator file>
    n=$(mean_of "$2" | awk '{print $1}')
    d=$(mean_of "$3" | awk '{print $1}')
    case "$n$d" in
        *no*) echo "  $1: an arm is missing" ; return ;;
    esac
    awk -v n="$n" -v d="$d" -v l="$1" \
        'BEGIN { printf "  %-46s %6.3f  (%+.1f %%)\n", l, n/d, (n/d - 1) * 100 }'
}

ratio "B over A, what the current program costs" "$SCRATCH/B.times" "$SCRATCH/A.times"
ratio "C over B, what 32 threads per block costs" "$SCRATCH/C.times" "$SCRATCH/B.times"
ratio "D over C, what a warm clock is worth" "$SCRATCH/D.times" "$SCRATCH/C.times"
ratio "E over A, all pins against the original" "$SCRATCH/E.times" "$SCRATCH/A.times"
ratio "B over E, what the run-time parameters cost" "$SCRATCH/B.times" "$SCRATCH/E.times"
ratio "F over E, what the other three pins add" "$SCRATCH/F.times" "$SCRATCH/E.times"
ratio "D over A, the swept number against the original" "$SCRATCH/D.times" "$SCRATCH/A.times"
echo
echo "  B over A is the figure the comparison rests on. On the H200 it measured"
echo "  3.1 percent, on the A100 73.3 percent, both on 2026-09-09, so the two"
echo "  cards do not carry the same program cost and every cross-chip ratio is"
echo "  inflated by the difference between them."
echo
echo "  E says whether that difference is the run-time parameters: the"
echo "  original's problem size is known to its compiler and the current"
echo "  program's is not, so the payload loop's trip count and the detection"
echo "  threshold are constants in one and registers in the other. E landing"
echo "  near A means that is the whole story."
echo
echo "  F then says how much of the fix is needed. All four pins together fix a"
echo "  binary to one point of the sweep, which is sixteen binaries per card."
echo "  The payload length alone is two, because it is the only pin that is a"
echo "  loop bound and the sweep uses two payload lengths. F near E means the"
echo "  cheap fix is the whole fix."
echo
echo "  Nothing here is appended to any csv. Copy this output into the notes."
