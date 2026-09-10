#!/usr/bin/env bash
#
# Decide why a regex sweep on this card came out slower than an earlier one:
# because of how the early exit is compiled, or because something else was
# using the GPU while it ran.
#
# The two explanations look identical in a results file. A results file records
# the configuration, and the configuration was the same; it records nothing
# about what else the machine was doing. So the slowdown has to be reproduced
# on purpose, with the two candidate causes separated, rather than argued about
# from the numbers that already exist.
#
# The two arms differ only in how the early-exit predicate reaches the kernel:
#
#   run time     the plain build. The predicate is a kernel argument, so the
#                compiler cannot prove the unrolled inner loop runs to
#                completion and keeps the row registers live across every
#                possible exit.
#   compile time built with -DPIN_EXIT_FIRST=1, which turns that argument into
#                a constant inside the kernel. The unroll's register
#                allocation is then free to assume the loop finishes.
#
# One source builds both, so nothing is checked out and no revision has to be
# named here. This is the same mechanism the sweep-time behaviour uses, so the
# arms bracket the change that is in question.
#
# The arms are interleaved rather than run one after the other, and their order
# alternates. A neighbouring job that comes and goes would otherwise land
# entirely on whichever arm happened to be running, and would read exactly like
# a property of the code; a load that drifts one way over the session would do
# the same to whichever arm always went first. Interleaved and alternated, a
# busy machine slows both arms together and the ratio survives, while a real
# code difference moves the ratio and holds it steady across reps.
#
# What else is on the card is also recorded, before and after, because a
# neighbour that was there for the whole run is invisible to the interleaving.
#
# Usage, from the root of the clone:
#
#   ./debug/check_regex_exit.sh a100
#   ./debug/check_regex_exit.sh h200
#
# It writes no csv and appends to nothing. Its two binaries are removed at the
# end, and they carry names of their own, so a sweep binary sitting beside them
# is neither used nor overwritten.

set -u

MACHINE=${1:-}
REPS=${REPS:-4}   # even, so each arm goes first equally often
SIGS=${SIGS:-10000000}
PAY=${PAY:-1024}
LEN=${LEN:-32}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-1}

case "$MACHINE" in
    a100) ARCHES=${ARCHES:-80}; TAG=a100; GPU_MATCH=A100 ;;
    h200) ARCHES=${ARCHES:-90}; TAG=h200; GPU_MATCH=H200 ;;
    *)
        echo "Usage: $0 <a100|h200>" >&2
        exit 2
        ;;
esac

SRC=App2/smith_waterman_dpi.cu
RT=App2/smith_waterman_dpi-$TAG-exit-rt
PIN=App2/smith_waterman_dpi-$TAG-exit-pin
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/regexexit.XXXXXX")

cleanup() { rm -rf "$SCRATCH" "$RT" "$PIN"; }
trap cleanup EXIT

name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$name" in
    *$GPU_MATCH*) echo "Node gpu: $name" ;;
    *)
        echo "This node reports '$name', not a $GPU_MATCH. Refusing to run." >&2
        exit 1
        ;;
esac

echo "Configuration: $SIGS signatures, ($PAY, $LEN), regex, intrinsic on,"
echo "  32 threads per block, exit at first report, $TRIALS trials after"
echo "  $WARMUP warm-up, $REPS reps of each arm, interleaved."
echo

# ---------------------------------------------------------------------------
# What else is using the card. A second job of any size changes what this one
# measures, and a job that was already running when the earlier sweep started
# is the explanation this check exists to rule in or out.
neighbours() {   # neighbours <when>
    echo "GPU occupancy, $1:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    local apps
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null)
    if [ -z "$apps" ]; then
        echo "    no compute processes"
    else
        printf '%s\n' "$apps" | sed 's/^/    /'
    fi
}

neighbours "before"
echo

# A foreign job anywhere on the node stops the run, rather than being noted and
# then measured through. Which physical card a foreign process sits on cannot be
# matched against the card this program picks without care, because CUDA and
# NVML order the devices differently, so the requirement here is the stricter
# and simpler one: no compute process on the node but this one. That is what a
# timing measurement needs in any case. Set ALLOW_BUSY=1 to measure anyway,
# which is worth doing only to show that a busy node is the cause of something.
foreign=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$foreign" -gt 0 ]; then
    if [ "${ALLOW_BUSY:-0}" = "1" ]; then
        echo "  $foreign compute process(es) already on this node. ALLOW_BUSY=1"
        echo "  is set, so the run continues, and every number below is a"
        echo "  measurement of a shared card rather than of this program."
        echo
    else
        echo "Refusing to run: $foreign compute process(es) are already using" >&2
        echo "this node's GPUs, and a timing measurement taken beside them" >&2
        echo "reports the sharing, not the program. Wait for the node, or set" >&2
        echo "ALLOW_BUSY=1 to measure anyway and say so wherever the number" >&2
        echo "is used." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "Build"
echo "=============================================================="
NVML=${NVML_LIBS:--lnvidia-ml -lpthread}
GEN="-gencode arch=compute_$ARCHES,code=sm_$ARCHES"

${NVCC:-nvcc} -O3 $GEN -Xptxas -v "$SRC" $NVML -o "$RT" \
    > "$SCRATCH/build.rt" 2>&1
${NVCC:-nvcc} -O3 $GEN -Xptxas -v -DPIN_EXIT_FIRST=1 "$SRC" $NVML -o "$PIN" \
    > "$SCRATCH/build.pin" 2>&1

for b in "$RT" "$PIN"; do
    if [ ! -x "$b" ]; then
        echo "Build failed for $b:" >&2
        tail -20 "$SCRATCH/build.$(basename "${b##*-}")" >&2
        exit 1
    fi
done
echo "Built both arms from $SRC."
echo

# ---------------------------------------------------------------------------
# Registers for the regex kernel each arm actually launches. If these are equal
# the two arms are the same machine code and no timing difference between them
# can come from the code.
WANT="regrows 1, regex 1,"

regs_from() {   # regs_from <build log> <label>
    awk -v want="$WANT" -v tag="$2" '
        /Function properties for/ {
            label = $0
            sub(/.*Function properties for /, "", label)
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
            if (index(label, want) == 0) next
            for (i = 1; i <= NF; i++) if ($i == "Used") u = $(i + 1)
            what = label
            sub(/^sw_scan</, "", what)
            sub(/>$/, "", what)
            printf "    %-14s %-52s %3d registers%s\n", tag, what, u, spill
        }' "$1"
}

echo "Registers for sm_$ARCHES, regex kernels with rows in registers:"
regs_from "$SCRATCH/build.rt"  "run time"
regs_from "$SCRATCH/build.pin" "compile time"
echo

# ---------------------------------------------------------------------------
# The per-trial rows the program writes to stdout begin with the trial number
# and carry the kernel time second. The column header it writes first is not a
# comment line, so rows are selected by shape rather than by not starting with
# a hash: a leading integer, then a number. Letting the header through puts a
# word into the sample, where awk scores it as zero and sorts it past every
# real reading.
run_arm() {   # run_arm <binary> <output file>
    ./"$1" --signatures "$SIGS" --payload "$PAY" --sig-len "$LEN" \
        --mode regex --rows registers --dpx on --block 32 --exit first \
        --trials "$TRIALS" --warmup "$WARMUP" 2>>"$SCRATCH/err" \
        | awk -F, '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]*\.?[0-9]+$/ { print $2 }' \
        >> "$2"
}

# The node was free when the run started, because the guard above insisted on
# it. It can stop being free at any point after that, so the count is taken
# again between arms and the largest is reported.
busy_peak=0
note_busy() {
    local n
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$n" -gt "$busy_peak" ] && busy_peak=$n
    return 0
}

echo "=============================================================="
echo "Timing"
echo "=============================================================="
: > "$SCRATCH/rt"
: > "$SCRATCH/pin"
: > "$SCRATCH/err"
for r in $(seq 1 "$REPS"); do
    # The order alternates. Interleaving alone is not enough when the load on
    # the node drifts one way over the session, because whichever arm always
    # goes first then always meets the lighter machine, and that bias reads as
    # a property of the code exactly like the effect being looked for.
    printf '  rep %d of %d ' "$r" "$REPS"
    if [ $((r % 2)) -eq 1 ]; then
        run_arm "$RT"  "$SCRATCH/rt";  note_busy; printf 'run-time done, '
        run_arm "$PIN" "$SCRATCH/pin"; note_busy; printf 'compile-time done\n'
    else
        run_arm "$PIN" "$SCRATCH/pin"; note_busy; printf 'compile-time done, '
        run_arm "$RT"  "$SCRATCH/rt";  note_busy; printf 'run-time done\n'
    fi
done
echo
# The count is taken between arms, when this check holds nothing on the card,
# so anything at all is somebody else's.
if [ "$busy_peak" -gt 0 ]; then
    echo "  WARNING: up to $busy_peak foreign compute process(es) appeared on"
    echo "  this node during the run, after the guard let it start, so the"
    echo "  readings below are of a shared card."
    echo
fi

# median, mean, smallest, largest, spread, count. The median leads for the
# reason the overhead control gives: a cold clock boosts for whole trials, so a
# short run of readings is not unimodal and its mean sits between the two
# modes.
stat_of() {
    awk '{ v[++n] = $1; s += $1 }
         END {
             if (n == 0) { print "no runs"; exit }
             m = s / n
             for (i = 2; i <= n; i++) {       # insertion sort, n is small
                 x = v[i]
                 for (j = i - 1; j >= 1 && v[j] > x; j--) v[j + 1] = v[j]
                 v[j + 1] = x
             }
             med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
             printf "%.6f %.6f %.6f %.6f %.1f %d",
                    med, m, v[1], v[n], (v[n] - v[1]) / m * 100, n
         }' "$1"
}

read -r rt_med rt_mean rt_min rt_max rt_spread rt_n <<EOF
$(stat_of "$SCRATCH/rt")
EOF
read -r pin_med pin_mean pin_min pin_max pin_spread pin_n <<EOF
$(stat_of "$SCRATCH/pin")
EOF

printf '%-14s %10s %10s %10s %10s %8s %4s\n' \
       "arm" "median" "mean" "min" "max" "spread" "n"
printf '%-14s %10.6f %10.6f %10.6f %10.6f %7.1f%% %4d\n' \
       "run time" "$rt_med" "$rt_mean" "$rt_min" "$rt_max" "$rt_spread" "$rt_n"
printf '%-14s %10.6f %10.6f %10.6f %10.6f %7.1f%% %4d\n' \
       "compile time" "$pin_med" "$pin_mean" "$pin_min" "$pin_max" "$pin_spread" "$pin_n"
echo
awk -v a="$rt_med" -v b="$pin_med" \
    'BEGIN { if (b > 0) printf "  run time over compile time: %.3f\n", a / b }'
echo

if [ -s "$SCRATCH/err" ]; then
    echo "Messages on stderr:"
    sed 's/^/    /' "$SCRATCH/err"
    echo
fi

neighbours "after"
echo

# ---------------------------------------------------------------------------
cat <<'EOF'
==============================================================
Reading this
==============================================================
Compare the run-time median above against the same configuration in the
results file the question is about. Three outcomes, and they mean different
things:

  The run-time arm here is as slow as the results file, and the two arms
  differ by about that much
      The code is the cause. The predicate form changes the generated code on
      this card and the register counts above say by how much. Take the faster
      form for this variant and re-sweep.

  The run-time arm here is as slow as the results file, and the two arms are
  about equal
      Neither predicate form is the cause, and the machine is slow now in the
      same way it was slow then. Look at the occupancy lines above, and at
      anything the node runs on a schedule.

  The run-time arm here is as fast as the earlier, faster results file
      Nothing reproduces, so the slow sweep was measured against a transient.
      That file records a busy machine rather than this program, and it has to
      be measured again before any ratio is taken from it.

The register counts bound what the code can be responsible for, whatever the
timings do. Equal counts settle it outright, since identical machine code
cannot run at two speeds for a reason inside the code. Counts that differ by a
few out of a hundred and fifty, with neither arm spilling to local memory,
change occupancy by at most one warp per scheduler and cannot produce a large
factor either. A large factor needs a large register move or a spill, and both
are printed above.
EOF
