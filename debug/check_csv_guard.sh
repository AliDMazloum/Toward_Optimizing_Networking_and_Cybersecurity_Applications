#!/usr/bin/env bash
#
# Exercise the header guard on a machine that can build, in the three cases
# that matter. A results csv is appended to across many runs, and a binary
# built from a different commit writes a different set of columns; appending
# under a header that describes the older set leaves every reader one field
# out of step and nothing reports it. The programs now compare the header they
# would write against the one already in the file and stop before starting.
#
# The guard is cheap to get subtly wrong, in ways a build does not catch: a
# constant that matches nothing refuses every append, and one that matches too
# loosely lets the original problem back in. So it is tested rather than
# assumed.
#
# Usage, from the root of the clone:
#
#   ./debug/check_csv_guard.sh <path to a built binary> [extra args]
#
# For example:
#
#   ./debug/check_csv_guard.sh App1/floyd_warshall_routing-a100 --cpu --nodes 200
#   ./debug/check_csv_guard.sh App2/smith_waterman_dpi-a100 --cpu --signatures 2000
#
# The cases run on the CPU path, so any build works and no GPU is needed. Give
# it the smallest configuration the program will accept: the point is the csv,
# not the measurement.

set -u

BIN=${1:-}
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "Usage: $0 <path to a built binary> [extra args]" >&2
    echo "The binary must exist and be executable." >&2
    exit 2
fi
shift

WORK=$(mktemp -d "${TMPDIR:-/tmp}/csvguard.XXXXXX")
CSV=$WORK/out.csv
pass=0
fail=0

report() {   # report <name> <expected> <actual>
    if [ "$2" = "$3" ]; then
        echo "  PASS  $1"
        pass=$((pass + 1))
    else
        echo "  FAIL  $1 (expected $2, got $3)"
        fail=$((fail + 1))
    fi
}

echo "Binary : $BIN"
echo "Extra  : $*"
echo "Work   : $WORK"
echo

# ---------------------------------------------------------------------------
echo "1. A file that does not exist yet is written, header and all"
"$BIN" "$@" --trials 1 --csv "$CSV" >"$WORK/1.log" 2>&1
report "run succeeds" 0 $?
if [ -f "$CSV" ]; then
    echo "  header written: $(head -1 "$CSV")"
    rows=$(($(wc -l < "$CSV") - 1))
    report "one data row" 1 "$rows"
else
    echo "  FAIL  no csv was written"
    fail=$((fail + 1))
fi
echo

# ---------------------------------------------------------------------------
echo "2. The same file is appended to, because the header is the one we write"
before=$(wc -l < "$CSV")
"$BIN" "$@" --trials 1 --csv "$CSV" >"$WORK/2.log" 2>&1
report "run succeeds" 0 $?
after=$(wc -l < "$CSV")
report "a row was added" "$((before + 1))" "$after"
echo

# ---------------------------------------------------------------------------
echo "3. A file whose header is one column short is refused, and left alone"
OLD=$WORK/old.csv
# Drop the last column from the header, which is what a binary built before a
# column existed leaves behind. The rows are copied unchanged, so the file is
# exactly the shape the guard has to catch.
head -1 "$CSV" | sed 's/,[^,]*$//' > "$OLD"
tail -n +2 "$CSV" >> "$OLD"
sum_before=$(cksum < "$OLD")

"$BIN" "$@" --trials 1 --csv "$OLD" >"$WORK/3.log" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
    echo "  FAIL  the run succeeded; the guard did not catch a short header"
    fail=$((fail + 1))
else
    echo "  PASS  the run stopped with status $status"
    pass=$((pass + 1))
fi
sum_after=$(cksum < "$OLD")
report "the file was not touched" "$sum_before" "$sum_after"
if grep -q "header" "$WORK/3.log"; then
    echo "  PASS  it said why:"
    sed 's/^/        /' "$WORK/3.log" | tail -8
    pass=$((pass + 1))
else
    echo "  FAIL  nothing in the output explains the refusal"
    fail=$((fail + 1))
fi
echo

# ---------------------------------------------------------------------------
echo "=============================================================="
echo "$pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
    rm -rf "$WORK"
    echo "The guard holds. Working files removed."
    exit 0
fi
echo "Logs and files kept in $WORK."
exit 1
