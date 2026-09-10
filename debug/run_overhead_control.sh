#!/usr/bin/env bash
#
# Measure what the current App2 program costs against the original one, on this
# machine, and find which run-time parameter accounts for the difference.
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
# size difference. Whatever the constants say is what every arm runs.
#
# Both programs are run as separate processes, one timed launch each, because
# the original has no trial loop and its warm-up is commented out: it measures
# a cold launch. Comparing that against a warm mean would charge the difference
# to the program instead of to the clock state. The first run of each arm is
# discarded, the rest are summarised.
#
# The arms:
#
#   A          original, cold single launch
#   B          current, cold single launch, 64 threads per block, same geometry
#   C          current, cold single launch, 32 threads per block, swept geometry
#   D          current, the swept protocol, five trials after one warm-up
#   one per    current, built with a PIN_ macro that turns a run-time parameter
#   pin set    into a compile-time constant, cold, 64 threads per block
#
# B over A is what the current program costs at matched geometry. C over B is
# what the block size costs. D against C is what a warm clock is worth. Each
# pin arm against A says whether that parameter is the cause.
#
# Usage, from the root of the clone:
#
#   ./debug/run_overhead_control.sh a100
#   ./debug/run_overhead_control.sh h200
#
# The original writes a signatures file of about 160 MB into its working
# directory on every run and does not remove it. Each arm runs in a scratch
# directory that is deleted at the end.

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
SRC=App2/smith_waterman_dpi.cu
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/overhead.XXXXXX")
ROOT=$(pwd)

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# The configuration, read from the original source so the arms cannot diverge.
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

# The detection threshold is floor(alpha * signature length) with the swept
# alpha of 0.8, so it is derived from a measured input rather than typed.
THRESHOLD=$(awk -v l="$LEN" 'BEGIN { printf "%d", int(0.8 * l) }')

# One entry per build: a label, a semicolon, the flags. "all" comes first so a
# session cut short still has the figure that matters most. The payload pin is
# not scanned by default: it was measured on the A100 on 2026-09-09 and changed
# neither the timing nor a single register count, so it is settled. Put it back
# by overriding PIN_SETS if that ever needs rechecking.
PIN_SETS=${PIN_SETS:-"all;-DPIN_SIGNATURES=$SIGS -DPIN_PAYLOAD=$PAY -DPIN_THRESHOLD=$THRESHOLD -DPIN_EXIT_FIRST=1
signatures;-DPIN_SIGNATURES=$SIGS
threshold;-DPIN_THRESHOLD=$THRESHOLD
exit-first;-DPIN_EXIT_FIRST=1"}

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
    echo "  NOTE this is a short run. The figures this control is compared"
    echo "       against were taken at 10,000,000 signatures, where the kernel"
    echo "       runs about a second on the A100 and cold-launch noise is a"
    echo "       small fraction of it. Below that the noise grows and the"
    echo "       numbers are not comparable with the earlier ones. Restore the"
    echo "       constants in the original source to compare against them."
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

# The original, built here rather than through the Makefile because it is not
# one of the programs the Makefile maintains. Same optimisation level and
# architecture as the current program; it needs neither NVML nor OpenMP.
${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES -Xptxas -v \
    "$OLD_SRC" -o "$OLD" > "$SCRATCH/build.old" 2>&1
if [ ! -x "$OLD" ]; then
    echo "The original did not build:" >&2
    cat "$SCRATCH/build.old" >&2
    exit 1
fi

# The unpinned current program, compiled again only to capture its register
# report; the binary the Makefile already produced is the one that runs.
${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES -Xptxas -v \
    "$SRC" ${NVML_LIBS:--lnvidia-ml -lpthread} -o /dev/null \
    > "$SCRATCH/build.none" 2>&1

printf '%s\n' "$PIN_SETS" | while IFS=';' read -r pinlabel pinflags; do
    [ -z "$pinlabel" ] && continue
    ${NVCC:-nvcc} -O3 -gencode arch=compute_$ARCHES,code=sm_$ARCHES -Xptxas -v \
        $pinflags "$SRC" ${NVML_LIBS:--lnvidia-ml -lpthread} \
        -o "App2/smith_waterman_dpi-$TAG-pin-$pinlabel" \
        > "$SCRATCH/build.$pinlabel" 2>&1
    if [ ! -x "App2/smith_waterman_dpi-$TAG-pin-$pinlabel" ]; then
        echo "Build failed for pin set $pinlabel:" >&2
        tail -20 "$SCRATCH/build.$pinlabel" >&2
    fi
done

echo "Built the original, the current program, and one binary per pin set."
echo

# ---------------------------------------------------------------------------
# Registers and spills, read out of the builds themselves rather than from a
# second round of compiles. A kernel that spills to local memory is slower for
# a reason no timing can show. The current program instantiates its kernel
# thirty-two times, so the mangled template arguments are translated and only
# the instantiations worth looking at are printed.
#
# Both signature lengths are printed, not only the one this control times. The
# sweep runs (512, 16) and (1024, 32), and the longer signature is much the
# heavier kernel: on the A100 it compiled to 237 registers against 126, which
# is about 13 percent occupancy, so it stands to gain more from anything that
# frees registers, and no arm here times it. Printing both means a rebuild
# shows whether it gained.
WANT="dpx 1, regrows 1,"

regs_from() {   # regs_from <build log> <label>
    awk -v want="$WANT" -v tag="$2" '
        /Function properties for/ {
            label = $0
            sub(/.*Function properties for /, "", label)
            # Five template parameters since the early-exit flag joined them:
            # sw_scan<signature length, intrinsic, rows in registers, regex,
            # exit at first report>. The last one must appear in the label or
            # its two instantiations print as indistinguishable duplicates.
            if (match(label, /_Z7sw_scanILi[0-9]+ELb[01]ELb[01]ELb[01]ELb[01]E/)) {
                a = substr(label, RSTART, RLENGTH)
                gsub(/[^0-9]/, " ", a)
                split(a, f, " ")
                label = sprintf("sw_scan<len %s, dpx %s, regrows %s, regex %s, exit %s>",
                                f[2], f[3], f[4], f[5], f[6])
            }
            spill = ""
            next
        }
        /spill stores/ {
            if ($0 !~ /0 bytes spill stores/) spill = "  SPILLS"
            next
        }
        /Used [0-9]+ registers/ {
            if (want != "" && index(label, want) == 0) next
            for (i = 1; i <= NF; i++) if ($i == "Used") u = $(i + 1)
            what = label
            sub(/^sw_scan<len /, "", what)
            sub(/, dpx [01], regrows [01], regex [01], exit /, " exit ", what)
            sub(/>$/, "", what)
            printf "    %-22s len %-10s %3d registers%s\n", tag, what, u, spill
        }' "$1"
}

echo "Registers for sm_$ARCHES, for the one kernel this control runs:"
echo "  $WANT"
awk '/Used [0-9]+ registers/ {
         for (i = 1; i <= NF; i++) if ($i == "Used") u = $(i + 1)
         printf "    %-22s %3d registers  (its only kernel)\n", "original", u
     }' "$SCRATCH/build.old"
regs_from "$SCRATCH/build.none" "no pins"
printf '%s\n' "$PIN_SETS" | while IFS=';' read -r pinlabel pinflags; do
    [ -z "$pinlabel" ] && continue
    [ -f "$SCRATCH/build.$pinlabel" ] && regs_from "$SCRATCH/build.$pinlabel" "pin $pinlabel"
done
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
             for (i = 2; i <= n; i++) {       # insertion sort, n is small
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
        printf "  %-34s did not run\n" "$1"
        return
    fi
    printf "  %-34s median %s  mean %s  range %s to %s  spread %s %%  n=%s\n" \
           "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}

ratio() {   # ratio <label> <numerator file> <denominator file>
    n=$(mean_of "$2" | awk '{print $1}')
    d=$(mean_of "$3" | awk '{print $1}')
    case "$n$d" in
        *no*) printf "  %-46s an arm is missing\n" "$1"; return ;;
    esac
    awk -v n="$n" -v d="$d" -v l="$1" \
        'BEGIN { printf "  %-46s %6.3f  (%+.1f %%)\n", l, n/d, (n/d - 1) * 100 }'
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

# new_arm <outfile> <block> <trials> <warmup> [binary]
new_arm() {
    out=$1; block=$2; trials=$3; warmup=$4; bin=${5:-$NEW}
    : > "$out"
    if [ ! -x "$bin" ]; then
        echo "  $bin was not built; the arm is skipped." >&2
        return
    fi
    for i in $(seq 1 $REPS); do
        # No --verify here, so every arm runs with the same default the sweep
        # ran with. Verification happens on the host after the kernel and is
        # outside the timed window in any case.
        t=$(./"$bin" --signatures $SIGS --payload $PAY --sig-len $LEN \
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
new_arm "$SCRATCH/B.times" 64 1 0
echo

echo "=============================================================="
echo "Arm C: the current program, cold single launch, 32 threads per block"
echo "=============================================================="
new_arm "$SCRATCH/C.times" 32 1 0
echo

echo "=============================================================="
echo "Arm D: the current program, the swept protocol"
echo "=============================================================="
new_arm "$SCRATCH/D.times" 32 5 1
echo

printf '%s\n' "$PIN_SETS" | while IFS=';' read -r pinlabel pinflags; do
    [ -z "$pinlabel" ] && continue
    echo "=============================================================="
    echo "Pin arm '$pinlabel': cold single launch, 64 threads per block"
    echo "  built with $pinflags"
    echo "=============================================================="
    new_arm "$SCRATCH/pin.$pinlabel.times" 64 1 0 \
        "App2/smith_waterman_dpi-$TAG-pin-$pinlabel"
    echo
done

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "What happened"
echo "=============================================================="
echo "  Every ratio below is taken on the medians, for the reason recorded"
echo "  above the statistic."
echo
report "A original, cold, block 64" "$SCRATCH/A.times"
report "B current, cold, block 64" "$SCRATCH/B.times"
report "C current, cold, block 32" "$SCRATCH/C.times"
report "D current, swept protocol" "$SCRATCH/D.times"
printf '%s\n' "$PIN_SETS" | while IFS=';' read -r pinlabel pinflags; do
    [ -z "$pinlabel" ] && continue
    report "pin $pinlabel, cold, block 64" "$SCRATCH/pin.$pinlabel.times"
done
echo

ratio "B over A, what the current program costs" "$SCRATCH/B.times" "$SCRATCH/A.times"
ratio "C over B, what 32 threads per block costs" "$SCRATCH/C.times" "$SCRATCH/B.times"
ratio "D over C, what a warm clock is worth" "$SCRATCH/D.times" "$SCRATCH/C.times"
printf '%s\n' "$PIN_SETS" | while IFS=';' read -r pinlabel pinflags; do
    [ -z "$pinlabel" ] && continue
    ratio "pin $pinlabel over A, against the original" \
          "$SCRATCH/pin.$pinlabel.times" "$SCRATCH/A.times"
done
echo

echo "  B over A is the figure the comparison rests on. It measured 3.1 percent"
echo "  on the H200 and 78.4 percent on the A100 before the early-exit flag was"
echo "  made an architecture choice, and the cause was traced to that flag"
echo "  alone: pinning it took the A100 from 126 registers to 79 and from"
echo "  1.1307 s to 0.6283 s, while pinning the signature count, the payload"
echo "  length or the threshold changed nothing."
echo
echo "  So on a build carrying that change, B over A should now read near zero"
echo "  on the A100, its register line should read 79 rather than 126, and the"
echo "  H200 should be exactly where it was, because only architectures below"
echo "  sm_90 take the compile-time form. The pin arms should now differ from"
echo "  B by nothing on the A100, since the flag is already decided there."
echo "  Anything else means the change did not do what it was measured to do."
echo
echo "  Nothing here is appended to any csv. Copy this output into the notes."
