#!/usr/bin/env bash
#
# Measure what the current App2 program costs against the original one, on this
# machine, at one configuration.
#
# The original program is Old_files/App2/DPI_v7.2.cu. Its problem size is fixed
# at compile time to 10,000,000 signatures, a 512 byte payload and 16 byte
# signatures, it uses the halfword intrinsic unconditionally, and it launches
# 64 threads per block over 5,000,000 threads. That is exactly one point of the
# current program's sweep, so the two can be compared directly there and only
# there.
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
OLD=Old_files/App2/DPI_v7.2-$TAG
SIGS=10000000
PAY=512
LEN=16
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/overhead.XXXXXX")
ROOT=$(pwd)

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
    Old_files/App2/DPI_v7.2.cu -o "$OLD" || exit 1
echo "Built $NEW and $OLD"
echo

# ---------------------------------------------------------------------------
# mean_of <file>   reads one number per line, drops the first, prints the mean,
#                  the smallest, the largest and the spread as a percentage.
mean_of() {
    awk 'NR > 1 { v[++n] = $1; s += $1 }
         END {
             if (n == 0) { print "no runs"; exit }
             m = s / n
             lo = hi = v[1]
             for (i = 1; i <= n; i++) {
                 if (v[i] < lo) lo = v[i]
                 if (v[i] > hi) hi = v[i]
             }
             printf "%.6f %.6f %.6f %.2f %d", m, lo, hi, (hi - lo) / m * 100, n
         }' "$1"
}

report() {   # report <label> <file>
    set -- "$1" $(mean_of "$2")
    if [ "$2" = "no" ]; then
        printf "  %-38s did not run\n" "$1"
        return
    fi
    printf "  %-38s mean %s s   range %s to %s   spread %s %%   n=%s\n" \
           "$1" "$2" "$3" "$4" "$5" "$6"
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
# new_arm <label> <outfile> <block> <trials> <warmup>
new_arm() {
    label=$1; out=$2; block=$3; trials=$4; warmup=$5
    : > "$out"
    for i in $(seq 1 $REPS); do
        # No --verify here, so every arm runs with the same default the sweep
        # ran with. Verification happens on the host after the kernel and is
        # outside the timed window in any case.
        t=$(./$NEW --signatures $SIGS --payload $PAY --sig-len $LEN \
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

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "What happened"
echo "=============================================================="
report "A original, cold, block 64" "$SCRATCH/A.times"
report "B current, cold, block 64" "$SCRATCH/B.times"
report "C current, cold, block 32" "$SCRATCH/C.times"
report "D current, swept protocol" "$SCRATCH/D.times"
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
ratio "D over A, the swept number against the original" "$SCRATCH/D.times" "$SCRATCH/A.times"
echo
echo "  B over A is the figure the comparison rests on. On the H200 it measured"
echo "  3.2 percent. A number of that order here means the two programs measure"
echo "  the same thing on this card and nothing needs a caveat; a large one"
echo "  means this card's numbers carry the current program's cost, not the"
echo "  card's, and every cross-chip ratio has to say so."
echo
echo "  Nothing here is appended to any csv. Copy this output into the notes."
