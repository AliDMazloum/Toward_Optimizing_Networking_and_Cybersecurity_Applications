#!/usr/bin/env bash
#
# Build a machine's binaries, characterise its power instrument, and run the
# two energy sweeps, in that order, so a node can be handed the whole job in
# one command when access to it opens up.
#
# The characterisation step is not a formality. Two of its outputs decide how
# the energy may be reported afterwards:
#
#   the idle draw of the card the kernel actually uses, which is the floor every
#   short run reports and the baseline load draw has to be read against; and
#
#   how often the driver refreshes its power reading, which sets the shortest
#   run whose energy means anything. A run finishing inside one refresh reports
#   a value the driver computed before it started. That interval differs by
#   machine and driver, so it is measured here rather than assumed.
#
# Usage, from the root of the clone on the machine that has the GPU:
#
#   ./debug/run_energy_sweeps.sh a100            build, characterise, sweep
#   ./debug/run_energy_sweeps.sh h200
#   ./debug/run_energy_sweeps.sh h200 --check-only    stop after characterising
#
# On a node shared with other users, name a free card first, because a sweep now
# refuses to run on a card somebody else is computing on:
#
#   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
#   SWEEP_GPU=3 ./debug/run_energy_sweeps.sh a100
#
# The variable reaches the sweeps through the environment and needs nothing
# added here. Energy follows it too, because both programs bind their power
# sampler by PCI bus id taken from the CUDA device rather than by an index.
#
# Sweeps append, and the workbook keeps the last five trials of each
# configuration, so re-running after an interruption does not duplicate work.

set -u

TAG=${1:-}
MODE=${2:-}

case "$TAG" in
    a100) ARCHES=80; GPU_MATCH=A100 ;;
    h200) ARCHES=90; GPU_MATCH=H200 ;;
    *)
        echo "Usage: $0 <a100|h200> [--check-only]" >&2
        exit 2
        ;;
esac

APP1=App1/floyd_warshall_routing-$TAG
APP2=App2/smith_waterman_dpi-$TAG
CHECK_NODES=${CHECK_NODES:-24000}

# How long the App2 timed window must be before the two energy instruments
# agree. Both cards refresh their sampled power and step their energy counter
# about every 100 ms, so a window of twenty steps holds the quantisation near
# five percent, which is the bound the analysis applies. Two seconds is that
# twenty steps. It is not a per-machine constant and it is not the run length:
# the program divides every reported figure back down to one scan, and writes
# the number of scans it did into the repeat column.
#
# App1 cannot use this. Floyd-Warshall converges its matrix in place, so a
# second pass over the same matrix is not the same work, and restoring the
# matrix would put the copy inside the timed window.
MIN_WINDOW=${MIN_WINDOW:-2.0}
LOG=$(mktemp -d "${TMPDIR:-/tmp}/energy-$TAG.XXXXXX")

echo "Tag $TAG, arch $ARCHES, logs in $LOG"
echo

# A csv written before the programs gained a column cannot be appended to: the
# new rows would carry more fields than its header describes, and nothing
# downstream would notice. The programs themselves now compare the header they
# would write against the one already in the file and stop before the run, for
# any column and any csv, so this script no longer carries its own narrower
# check. A sweep that stops on the first configuration with a message about
# headers has hit it; remove the named file and let the run write it afresh.
# `debug/check_csv_guard.sh` exercises that guard on a built binary.

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "1. Build"
echo "=============================================================="
make ARCHES=$ARCHES TAG=$TAG app1 app2 2>&1 | tee "$LOG/build.log"
if [ ! -x "$APP1" ] || [ ! -x "$APP2" ]; then
    echo "Build did not produce both binaries; stopping." >&2
    exit 1
fi
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "2. Characterise the power instrument"
echo "=============================================================="
./debug/validate_gpu_power.sh "$APP1" "$CHECK_NODES" "${DEVICE:-0}" \
    2>&1 | tee "$LOG/power_check.log"
echo

echo "--------------------------------------------------------------"
echo "The three lines that decide how energy may be reported:"
echo "--------------------------------------------------------------"
grep -E "busiest GPU during the run|mean interval between consecutive changes|program reported" \
    "$LOG/power_check.log" || echo "  (not found; read $LOG/power_check.log in full)"
echo
echo "Add the refresh interval to GPU_POWER_REFRESH_BY_CARD in validate_energy.py,"
echo "keyed on this card's name, before judging which of its energy points are"
echo "reportable. The intervals there were measured per card and none carries"
echo "over to another."
echo

if [ "$MODE" = "--check-only" ]; then
    echo "Stopping after the characterisation, as asked."
    exit 0
fi

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "3. App1 energy sweep"
echo "=============================================================="
make ARCHES=$ARCHES TAG=$TAG GPU_MATCH=$GPU_MATCH sweep \
    SWEEP_FLAGS="--layout tiled --store changed --sync per-launch --energy" \
    SWEEP_CSV=Results/app1_energy_$TAG.csv 2>&1 | tee "$LOG/app1_sweep.log"
APP1_STATUS=${PIPESTATUS[0]}
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "4. App2 energy sweep"
echo "=============================================================="
make ARCHES=$ARCHES TAG=$TAG GPU_MATCH=$GPU_MATCH sweep2 \
    SWEEP2_FLAGS="--mode literal --rows registers --energy --min-window $MIN_WINDOW" \
    SWEEP2_CSV=Results/app2_energy_repeat_$TAG.csv 2>&1 | tee "$LOG/app2_sweep.log"
APP2_STATUS=${PIPESTATUS[0]}
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "5. What happened"
echo "=============================================================="
for f in Results/app1_energy_$TAG.csv Results/app2_energy_repeat_$TAG.csv; do
    if [ -f "$f" ]; then
        rows=$(($(wc -l < "$f") - 1))
        bad=$(awk -F, 'NR==1 { for(i=1;i<=NF;i++) if($i=="mismatches") c=i; next }
                       c && $c != "0" { n++ } END { print n+0 }' "$f")
        echo "  $f: $rows rows, $bad with a nonzero mismatch"
    else
        echo "  $f: not written"
    fi
done
echo
echo "  App1 sweep exit status: $APP1_STATUS"
echo "  App2 sweep exit status: $APP2_STATUS"
if [ "$APP1_STATUS" -ne 0 ] || [ "$APP2_STATUS" -ne 0 ]; then
    echo
    echo "  A sweep stopped early. Both stop at the first failure rather than"
    echo "  leaving a half-written file that looks complete, so the tail of the"
    echo "  log in $LOG says what ended it. Re-running appends and does not"
    echo "  repeat the configurations that already finished."
fi
echo
echo "  Commit the CSVs from Results/, then on the workstation rerun"
echo "  build_results_workbook.py and validate_energy.py, the latter with this"
echo "  machine's refresh interval, before quoting any of it."
echo
echo "Logs are in $LOG."
