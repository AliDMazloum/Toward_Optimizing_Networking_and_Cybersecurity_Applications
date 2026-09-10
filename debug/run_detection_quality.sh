#!/usr/bin/env bash
#
# Measure detection quality: how often the detector fires on a payload that
# carries a planted signature, and how often it fires on one that does not,
# as a curve over the threshold alpha and over the alphabet the synthetic data
# is drawn from.
#
# Why the alphabet is a variable here rather than a constant. Two unrelated
# strings agree at one position with probability one over the alphabet size, so
# a false positive rate measured over 26 lowercase letters is roughly ten times
# the rate over the whole byte range, and neither number is a property of the
# detector on its own. Reporting one alphabet would state a result about the
# generator. Reporting three states what a change of traffic would do, which is
# what a real corpus would have told us and is the part that generalises.
#
# What one run of the program contributes: one payload, scanned against every
# signature, and one bit saying whether anything crossed the threshold. So each
# row below is a single Bernoulli trial and the rates come from repeating it
# over seeds. The resolution of a rate is one over the seed count, so a rate
# below that reads as zero and the seed count has to be quoted beside it.
#
# The alpha sweep cannot be collapsed. The scores do not depend on alpha but the
# decision does, and the program reports the first crossing rather than the
# largest score, so knowing that a payload crossed at a low threshold says
# nothing about a high one. That is why this is a loop over alpha rather than
# one pass with the thresholds applied afterwards.
#
# Usage, from the root of the clone:
#
#   ./debug/run_detection_quality.sh h200
#   SWEEP_GPU=3 ./debug/run_detection_quality.sh a100
#   SEEDS=20 ./debug/run_detection_quality.sh h200      a short rehearsal
#
# Unlike the sweeps, this does not refuse a card another job is on, and the
# difference is deliberate: it measures which payloads are detected, not how
# long anything takes, and a detection is the same detection on a busy card. A
# neighbour costs wall-clock here and nothing else. SWEEP_GPU still pins the
# card, so the run can be put somewhere out of the way.
#
# It writes one file, Results/app2_detection_<tag>.csv, one row per run, and
# appends nothing to any existing measurement.

set -u

MACHINE=${1:-}
case "$MACHINE" in
    a100) ARCHES=${ARCHES:-80}; TAG=a100; GPU_MATCH=A100 ;;
    h200) ARCHES=${ARCHES:-90}; TAG=h200; GPU_MATCH=H200 ;;
    *)    echo "Usage: $0 <a100|h200>" >&2; exit 2 ;;
esac

# Every one of these is overridable, and each is a choice worth seeing rather
# than a constant buried in the loop.
#
#   SIGS      the smallest signature count the reported sweeps use. The false
#             positive rate rises with it, because every signature is another
#             chance to agree, so a rate here is a floor for the larger counts
#             and the count has to be quoted with it.
#   CONFIGS   both (payload, signature length) pairs the reported sweeps use.
#   ALPHAS    the reported threshold, 0.8, with a spread either side of it.
#             The threshold the kernel applies is floor(alpha * L), so alpha
#             steps finer than 1/L land on the same integer and buy nothing:
#             at length 16 the distinct thresholds are 8/16 through 16/16.
#             A grid in steps of 1/32 resolves both signature lengths, at
#             the cost of running longer.
#   SEEDS     trials per point. The rate resolution is 1/SEEDS.
#   PLANT     which signature is planted; any index below SIGS will do.
SIGS=${SIGS:-10000}
CONFIGS=${CONFIGS:-"512x16 1024x32"}
ALPHAS=${ALPHAS:-"0.50 0.60 0.70 0.80 0.90 0.95"}
SEEDS=${SEEDS:-100}
ALPHABETS=${ALPHABETS:-"lower26 ascii95 bytes256"}
MODES=${MODES:-"literal regex"}
PLANT=${PLANT:-1356}

BIN=App2/smith_waterman_dpi-$TAG
# Overridable, because the useful shape of this experiment is two runs
# rather than one: a wide alpha grid at moderate seed count to show where
# the detector turns over, and a deep run at the reported threshold alone
# to bound the false positive rate there. Those want separate files.
OUT=${OUT:-Results/app2_detection_$TAG.csv}

name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$name" in
    *$GPU_MATCH*) echo "Node gpu: $name" ;;
    *) echo "This node reports '$name', not a $GPU_MATCH. Refusing to run." >&2
       exit 1 ;;
esac

if [ -n "${SWEEP_GPU:-}" ]; then
    uuid=$(nvidia-smi -i "$SWEEP_GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
    if [ -z "$uuid" ]; then
        echo "No GPU with index $SWEEP_GPU on this node." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$uuid
    echo "Pinned to gpu $SWEEP_GPU, $uuid"
fi

make ARCHES=$ARCHES TAG=$TAG app2 || exit 1
[ -x "$BIN" ] || { echo "Build did not produce $BIN." >&2; exit 1; }

if [ -e "$OUT" ]; then
    echo "$OUT already exists. Move it aside first; this script does not append," >&2
    echo "because a rerun with different settings would be indistinguishable from" >&2
    echo "more trials of the same ones." >&2
    exit 1
fi
mkdir -p Results || exit 1

runs=0
for a in $ALPHABETS; do for m in $MODES; do for c in $CONFIGS; do
    for al in $ALPHAS; do runs=$((runs + SEEDS * 2)); done
done; done; done
echo "$runs runs: $(echo $ALPHABETS | wc -w) alphabets, $(echo $MODES | wc -w) modes,"
echo "  $(echo $CONFIGS | wc -w) configurations, $(echo $ALPHAS | wc -w) alphas,"
echo "  $SEEDS seeds, planted and unplanted. At roughly a third of a second each"
echo "  that is about $((runs / 3 / 60)) minutes."
echo "Writing $OUT"
echo

printf 'alphabet,mode,payload,sig_len,signatures,alpha,seed,planted,found,report_sig,report_score,report_pos,mismatches\n' > "$OUT"

done_runs=0
for alphabet in $ALPHABETS; do
  for mode in $MODES; do
    for cfg in $CONFIGS; do
      pay=${cfg%x*}; len=${cfg#*x}
      for alpha in $ALPHAS; do
        for seed in $(seq 1 "$SEEDS"); do
          for planted in 1 0; do
            if [ "$planted" = 1 ]; then plant_arg="--plant $PLANT"; else plant_arg=""; fi
            row=$(./"$BIN" --signatures "$SIGS" --payload "$pay" --sig-len "$len" \
                    --mode "$mode" --rows registers --dpx on --block 32 \
                    --alphabet "$alphabet" --alpha "$alpha" --seed "$seed" \
                    $plant_arg --trials 1 --warmup 0 --verify report \
                    2>/dev/null \
                  | awk -F, '$1 ~ /^[0-9]+$/ { print $4","$5","$6","$7","$8 }' | head -1)
            [ -z "$row" ] && row=",,,,"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                   "$alphabet" "$mode" "$pay" "$len" "$SIGS" "$alpha" \
                   "$seed" "$planted" "$row" >> "$OUT"
            done_runs=$((done_runs + 1))
          done
        done
        printf '\r  %d of %d runs, at %s %s (%s, %s) alpha %s   ' \
               "$done_runs" "$runs" "$alphabet" "$mode" "$pay" "$len" "$alpha"
      done
    done
  done
done
echo
echo

# ---------------------------------------------------------------------------
# The confusion matrix, per point, straight out of the file, so a run that is
# interrupted still says what it found and so nobody has to reopen this in a
# notebook to see whether it worked.
#
# A planted payload that fires is a true positive and one that does not is a
# false negative; an unplanted payload that fires is a false positive and one
# that does not is a true negative. Nothing here is a per-signature rate: the
# unit is one payload scanned against every signature, which is the decision an
# operator actually sees.
awk -F, 'NR > 1 {
    key = $1 "," $2 "," $3 "x" $4 "," $6
    if ($8 == 1) { if ($9 == 1) tp[key]++; else fn[key]++ }
    else         { if ($9 == 1) fp[key]++; else tn[key]++ }
    seen[key] = 1
}
END {
    printf "%-9s %-8s %-9s %-6s %6s %6s %6s %6s %8s %8s\n",
           "alphabet", "mode", "config", "alpha", "TP", "FN", "FP", "TN", "TPR", "FPR"
    for (k in seen) {
        split(k, f, ",")
        pos = tp[k] + fn[k]; neg = fp[k] + tn[k]
        printf "%-9s %-8s %-9s %-6s %6d %6d %6d %6d %8s %8s\n",
               f[1], f[2], f[3], f[4], tp[k], fn[k], fp[k], tn[k],
               (pos ? sprintf("%.3f", tp[k] / pos) : "-"),
               (neg ? sprintf("%.3f", fp[k] / neg) : "-")
    }
}' "$OUT" | (read -r header; echo "$header"; sort)

cat <<EOF

Reading this
============
The true positive rate is a property of what was planted, and the two modes
plant different things, so they answer different questions.

In literal mode the planted signature's own text is copied into the payload and
every one of its characters scores, so a perfect plant reaches its full length.
The test is strict against floor(alpha * L), so it is detected at every alpha
below 1.0 and the true positive rate should be 1 throughout. A false negative
below alpha 1.0 means something is wrong, not that the threshold was strict.

That was not always so. The payload loop used to start at the second byte, so
the plant could reach only L-1 and nothing cleared alpha 0.95 at length 16. If
this table shows a true positive rate under 1 at a high alpha, check that the
binary is newer than that change rather than looking for a cause in the data.

In regex mode the planted signature contains wildcards, and the scoring rule
requires a signature with any wildcard to reach its full literal score. Alpha
does not enter that test at all, so the true positive rate is 1 at every alpha
by construction. That is worth reporting rather than hiding: it says the
formulation makes a wildcard signature's detection threshold-independent, and
the alpha curve in regex mode is a false positive curve.

Both rates are per payload against $SIGS signatures. The false positive rate
rises with the signature count, because each signature is another chance to
agree, so quote the count with the rate. The rate resolution is one over the
seed count, $SEEDS here, and a rate below that reads as zero rather than as
absent.
EOF
