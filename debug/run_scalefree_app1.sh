#!/usr/bin/env bash
#
# Sweep App1 over the generated graph on one machine, unattended.
#
# The evaluated graph moves from the chain to Barabasi-Albert preferential
# attachment. The chain runs are kept and become the validation set, because the
# chain has a closed-form distance oracle and a generated graph does not, so the
# stronger correctness claim lives there.
#
# Usage, from the root of the clone, one argument naming the machine:
#
#   ./debug/run_scalefree_app1.sh h200   the H200 and the EPYC 9355 hosting it
#   ./debug/run_scalefree_app1.sh a100   the A100
#   ./debug/run_scalefree_app1.sh epyc   the EPYC 7302P, timing and energy
#
# On a node shared with other users, name a free card first, because a sweep now
# refuses to run on a card somebody else is computing on. The variable reaches
# the sweeps through the environment and needs nothing added here:
#
#   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
#   SWEEP_GPU=3 ./debug/run_scalefree_app1.sh a100
#
# Each stage is independent and a failure in one does not stop the rest, because
# this is meant to be started and left. The summary at the end says which stages
# produced a file and how many rows, so a partial session is obvious rather than
# silent. Everything is written to files that do not exist yet, so nothing is
# appended to an existing measurement.
#
# On the epyc machine, export the CUDA toolkit environment first, including
# NVML_LIBS pointing at the stubs, as the machine notes describe; the Makefile
# picks those up from the environment and this script does not guess at them.

set -u

MACHINE=${1:-}
LOG=$(mktemp -d "${TMPDIR:-/tmp}/scalefree.XXXXXX")
SIZES=${SIZES:-"1000 2000 3000 6000 12000 24000"}
TRIALS=${TRIALS:-5}
WARMUP=${WARMUP:-1}

# The measured protocol for every CPU run in this study.
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}

case "$MACHINE" in
    h200) ARCHES=${ARCHES:-90}; TAG=h200; GPU_MATCH=H200; THREADS="32 16" ;;
    a100) ARCHES=${ARCHES:-80}; TAG=a100; GPU_MATCH=A100; THREADS="" ;;
    epyc) ARCHES=${ARCHES:-90}; TAG=epyc; GPU_MATCH="";   THREADS="16" ;;
    *)
        echo "Usage: $0 <h200|a100|epyc>" >&2
        exit 2
        ;;
esac

BIN=App1/floyd_warshall_routing-$TAG
STAGES=""

note() {   # note <name> <status>
    STAGES="$STAGES$1 $2
"
}

echo "Machine $MACHINE, tag $TAG, logs in $LOG"
echo "Sizes: $SIZES, $TRIALS trials after $WARMUP warm-up"
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "Build"
echo "=============================================================="
make ARCHES=$ARCHES TAG=$TAG app1 2>&1 | tee "$LOG/build.log"
if [ ! -x "$BIN" ]; then
    echo "Build did not produce $BIN; nothing else can run." >&2
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# The GPU sweeps. One file per machine carries both timing and energy: the
# sampling thread was measured not to perturb the timing, so a separate
# timing-only sweep would only be a second thing to keep in step.
if [ -n "$GPU_MATCH" ]; then
    echo "=============================================================="
    echo "Tiled sweep, with energy"
    echo "=============================================================="
    make ARCHES=$ARCHES TAG=$TAG GPU_MATCH=$GPU_MATCH sweep \
        SWEEP_NODES="$SIZES" SWEEP_TRIALS=$TRIALS SWEEP_WARMUP=$WARMUP \
        SWEEP_FLAGS="--layout tiled --store changed --sync per-launch --topology scale-free --energy" \
        SWEEP_CSV=Results/app1_scalefree_energy_$TAG.csv 2>&1 | tee "$LOG/tiled.log"
    note "tiled+energy" "${PIPESTATUS[0]}"
    echo

    echo "=============================================================="
    echo "Flat coalesced sweep, the layout comparison"
    echo "=============================================================="
    make ARCHES=$ARCHES TAG=$TAG GPU_MATCH=$GPU_MATCH sweep \
        SWEEP_NODES="$SIZES" SWEEP_TRIALS=$TRIALS SWEEP_WARMUP=$WARMUP \
        SWEEP_FLAGS="--layout coalesced --store changed --sync per-launch --topology scale-free" \
        SWEEP_CSV=Results/app1_scalefree_flat_$TAG.csv 2>&1 | tee "$LOG/flat.log"
    note "flat" "${PIPESTATUS[0]}"
    echo
fi

# ---------------------------------------------------------------------------
# The CPU sweeps, looped here rather than through the sweep target, because
# that target runs each size twice for the two DPX arms and the DPX flag means
# nothing on the CPU path.
#
# --no-verify is deliberate. On a generated graph the reference and the
# measurement are the same host triple loop on the same input, so the check
# would compare it against itself. The chain CPU runs already validate that
# loop against the closed form, so nothing is given up here.
for th in $THREADS; do
    echo "=============================================================="
    echo "CPU sweep, $th threads"
    echo "=============================================================="
    ENERGY=""
    if [ "$MACHINE" = "epyc" ]; then ENERGY="--energy"; fi
    # The loop runs inside a pipeline, so its exit status is set in a subshell
    # and has to come back through a file rather than a variable.
    rm -f "$LOG/cpu_$th.status"
    {
        for n in $SIZES; do
            echo "# $th threads, $n nodes"
            OMP_NUM_THREADS=$th ./$BIN --cpu --nodes "$n" --topology scale-free \
                --no-verify $ENERGY --trials $TRIALS --warmup $WARMUP \
                --csv Results/app1_scalefree_cpu_$TAG.csv \
                || { echo $? > "$LOG/cpu_$th.status"; break; }
        done
    } 2>&1 | tee "$LOG/cpu_$th.log"
    note "cpu-$th" "$(cat "$LOG/cpu_$th.status" 2>/dev/null || echo 0)"
    echo
done

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "What happened"
echo "=============================================================="
printf '%s' "$STAGES" | while read -r name status; do
    [ -z "$name" ] && continue
    if [ "$status" = "0" ]; then
        printf "  %-14s finished\n" "$name"
    else
        printf "  %-14s STOPPED, exit %s\n" "$name" "$status"
    fi
done
echo
for f in Results/app1_scalefree_energy_$TAG.csv \
         Results/app1_scalefree_flat_$TAG.csv \
         Results/app1_scalefree_cpu_$TAG.csv; do
    if [ -f "$f" ]; then
        rows=$(($(wc -l < "$f") - 1))
        printf "  %-44s %4d rows\n" "$f" "$rows"
    fi
done
echo
echo "  A stage that stopped left the rows it had already written, which are"
echo "  sound; the tail of its log in $LOG says what ended it, and re-running"
echo "  appends rather than repeating what finished."
echo
echo "  Commit the new files under Results/ and push."
