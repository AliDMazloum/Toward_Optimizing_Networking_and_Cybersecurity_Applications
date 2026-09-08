#!/usr/bin/env bash
#
# Check what the NVML power path in these programs actually measures.
#
# The energy these programs report comes from a sampling thread that calls
# nvmlDeviceGetPowerUsage while the kernel runs. Three separate faults make
# that number wrong in a way the number itself cannot reveal:
#
#   1. The wrong device. The programs pass one index to both cudaSetDevice and
#      nvmlDeviceGetHandleByIndex. CUDA orders devices by speed unless
#      CUDA_DEVICE_ORDER is PCI_BUS_ID, while NVML always orders them by PCI
#      bus id, so where a machine has more than one GPU the two orders can
#      disagree and the power is read from a card that is not running anything.
#   2. A driver value refreshed far more slowly than the sampler polls, so one
#      stale reading is integrated over and over.
#   3. Nothing wrong at all: a kernel that leaves the device far below its
#      power limit genuinely draws little.
#
# All three produce the same symptom, power that barely moves as the problem
# grows, and they call for opposite responses. This script separates them by
# watching one run from outside with nvidia-smi and comparing that against what
# the program reports for the same window.
#
# It writes nothing outside its own temporary directory and changes no state on
# the machine. It runs one job and watches it.
#
# Usage, from the root of the clone on a machine that has the GPU:
#
#   ./validate_gpu_power.sh [binary] [nodes] [device]
#
# Defaults: App1/floyd_warshall_routing-a100, 24000 nodes, device 0. Pass the
# same binary, size and device index as the run being checked, for example
# App1/floyd_warshall_routing-h200 on an H200 node.

set -u

BIN=${1:-App1/floyd_warshall_routing-a100}
NODES=${2:-24000}
DEV=${3:-0}
IDLE_SECONDS=${IDLE_SECONDS:-15}

if [ ! -x "$BIN" ]; then
    echo "Binary not found or not executable: $BIN" >&2
    echo "Build it first, for example: make a100-app1" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not on PATH; this script needs it as the second opinion" >&2
    exit 1
fi

OUT=$(mktemp -d "${TMPDIR:-/tmp}/gpupower.XXXXXX")
echo "Working directory: $OUT"
echo

QUERY=index,power.draw,utilization.gpu
SMI_FIELDS=--format=csv,noheader,nounits

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "1. What the machine says it has"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit,memory.total \
           --format=csv
echo
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA_DEVICE_ORDER    = ${CUDA_DEVICE_ORDER:-unset}"
echo
echo "The programs pass one index to both cudaSetDevice and"
echo "nvmlDeviceGetHandleByIndex. CUDA orders devices by speed unless"
echo "CUDA_DEVICE_ORDER is PCI_BUS_ID, while NVML always orders by PCI bus id."
echo "On a machine with more than one GPU those two orders can disagree, and"
echo "then the power is read from the wrong card. Section 4 tests that directly"
echo "rather than reasoning about it."
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "2. Idle draw, ${IDLE_SECONDS} s with nothing of ours running"
echo "=============================================================="
nvidia-smi --query-gpu=$QUERY $SMI_FIELDS -l 1 > "$OUT/idle.log" 2>/dev/null &
IDLE_PID=$!
sleep "$IDLE_SECONDS"
kill $IDLE_PID 2>/dev/null
wait $IDLE_PID 2>/dev/null

awk -F', *' '{ n[$1]++; p[$1]+=$2; if($3>u[$1]) u[$1]=$3 }
     END { for (i in n) printf "  GPU %s: idle mean %.1f W, peak utilisation %d%%, %d samples\n", i, p[i]/n[i], u[i]+0, n[i] }' \
     "$OUT/idle.log" | sort
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "3. Driver energy counter before the run"
echo "=============================================================="
nvidia-smi -q -d POWER > "$OUT/power_before.txt" 2>/dev/null
if grep -q "Total Energy Consumption" "$OUT/power_before.txt"; then
    grep "Total Energy Consumption" "$OUT/power_before.txt"
    echo "  (this counter is independent of the sampling loop, so the two can be compared)"
else
    echo "  the driver does not expose Total Energy Consumption here;"
    echo "  the cross-check in section 5 will be skipped"
fi
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "4. One measured run, watched from outside"
echo "=============================================================="
echo "Running: $BIN --nodes $NODES --layout tiled --trials 1 --warmup 0 --energy --device $DEV"
echo

nvidia-smi --query-gpu=$QUERY $SMI_FIELDS -l 1 > "$OUT/run.log" 2>/dev/null &
RUN_PID=$!

START=$(date +%s.%N)
"$BIN" --nodes "$NODES" --layout tiled --trials 1 --warmup 0 --energy \
       --device "$DEV" --power-csv "$OUT/trace.csv" --csv "$OUT/row.csv" \
       > "$OUT/stdout.txt" 2>&1
STATUS=$?
END=$(date +%s.%N)

kill $RUN_PID 2>/dev/null
wait $RUN_PID 2>/dev/null

if [ $STATUS -ne 0 ]; then
    echo "The run failed with status $STATUS. Its output:" >&2
    cat "$OUT/stdout.txt" >&2
    exit $STATUS
fi

echo "Wall time: $(awk -v a="$START" -v b="$END" 'BEGIN{printf "%.2f s", b-a}')"
echo
echo "What the program said about its device:"
grep -E "^# (nvml_device|cuda_device|device|gpu)" "$OUT/stdout.txt" || \
    echo "  (no device line in the header)"
echo
echo "Per GPU during the run, from nvidia-smi:"
awk -F', *' '{ n[$1]++; p[$1]+=$2; if($3>u[$1]) u[$1]=$3 }
     END { for (i in n) printf "  GPU %s: mean %.1f W, peak utilisation %d%%, %d samples\n", i, p[i]/n[i], u[i]+0, n[i] }' \
     "$OUT/run.log" | sort

BUSY=$(awk -F', *' '{ if($3>u[$1]) u[$1]=$3 }
       END { best=-1; for (i in u) if (u[i]+0>best) { best=u[i]+0; who=i } print who }' "$OUT/run.log")
BUSY_W=$(awk -F', *' -v g="$BUSY" '$1==g { n++; s+=$2 } END { if(n) printf "%.1f", s/n }' "$OUT/run.log")
IDLE_W=$(awk -F', *' -v g="$BUSY" '$1==g { n++; s+=$2 } END { if(n) printf "%.1f", s/n }' "$OUT/idle.log")
echo
echo "  busiest GPU during the run: index $BUSY (mean $BUSY_W W, idle was $IDLE_W W)"
echo

echo "What the program reported for the same window:"
if [ -s "$OUT/row.csv" ]; then
    head -1 "$OUT/row.csv"
    tail -1 "$OUT/row.csv"
    PROG_W=$(awk -F, 'NR==1 { for(i=1;i<=NF;i++) if($i=="mean_power_w") c=i }
             NR>1 && c { v=$c } END { print v }' "$OUT/row.csv")
    PROG_J=$(awk -F, 'NR==1 { for(i=1;i<=NF;i++) if($i=="energy_j") c=i }
             NR>1 && c { v=$c } END { print v }' "$OUT/row.csv")
else
    PROG_W=""; PROG_J=""
    echo "  the run wrote no csv row"
fi
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "5. How often the sampled value actually changed"
echo "=============================================================="
if [ -s "$OUT/trace.csv" ]; then
    awk -F, 'NR>1 {
        tot++
        if (!seen[$2]++) distinct++
        if (prev != "" && $2 != prev) {
            changes++
            # Time only between consecutive changes. The stretch before the
            # first change is not an interval; counting it would inflate the
            # answer by however long the value happened to sit still at the start.
            if (lastchange != "") { sum += $1 - lastchange; c++ }
            lastchange = $1
        }
        prev = $2
        if (min == "" || $2+0 < min) min = $2+0
        if ($2+0 > max) max = $2+0
    } END {
        printf "  %d samples, %d distinct values, %d changes\n", tot, distinct, changes+0
        printf "  range %.1f to %.1f W\n", min, max
        if (c > 0) printf "  mean interval between consecutive changes: %.1f ms (the sampler polls every 1 ms)\n", 1000*sum/c
        else print "  the value never changed for the whole run"
    }' "$OUT/trace.csv"
else
    echo "  no power trace was written"
fi
echo

nvidia-smi -q -d POWER > "$OUT/power_after.txt" 2>/dev/null
if grep -q "Total Energy Consumption" "$OUT/power_before.txt" 2>/dev/null; then
    echo "Driver energy counter, before and after:"
    grep "Total Energy Consumption" "$OUT/power_before.txt" | sed 's/^/  before:/'
    grep "Total Energy Consumption" "$OUT/power_after.txt"  | sed 's/^/  after :/'
    echo "  The difference covers the whole run plus this script's own gaps, so it"
    echo "  is an upper bound on the kernel's energy, not an equal comparison."
    echo
fi

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "6. Reading the result"
echo "=============================================================="
echo "  busiest GPU (nvidia-smi) : index $BUSY, mean $BUSY_W W during the run"
echo "  same GPU when idle       : $IDLE_W W"
echo "  program reported         : ${PROG_W:-n/a} W, ${PROG_J:-n/a} J"
echo
echo "  If the program's power is close to the busiest GPU's power, the reading"
echo "  is honest and the kernel genuinely draws little; the paper should then"
echo "  say the device is far from saturated and report idle draw beside it."
echo
echo "  If the busiest GPU drew much more than the program reported, the program"
echo "  is reading a different card or a stale value. Compare the busiest index"
echo "  above against the --device index that was passed ($DEV), and compare the"
echo "  bus id in the program's nvml_device line against the table in section 1."
echo "  The usual reason they disagree is CUDA_VISIBLE_DEVICES, shown in section"
echo "  1: it renumbers what CUDA sees while NVML goes on counting every card in"
echo "  the machine, so CUDA_DEVICE_ORDER=PCI_BUS_ID does not help. Any energy"
echo "  collected before the program resolved its handle by PCI bus id is void."
echo
echo "  If section 5 shows the value changing rarely or never, the driver is not"
echo "  refreshing it at the rate the sampler assumes, and the poll interval"
echo "  must be raised to the refresh interval before the energy means anything."
echo
echo "Everything is kept in $OUT (trace.csv, run.log, idle.log, stdout.txt)."
