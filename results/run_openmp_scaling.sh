#!/bin/bash
#
# results/run_openmp_scaling.sh
#
# Runs one OpenMP kernel binary across a fixed set of thread counts and
# appends each run's CSV line (the line bench_report_csv already prints
# to stdout) to a results file. Does NOT plot anything -- that's a
# separate script (results/plot.py), run only once the CSV has real
# data in it.
#
# Usage:
#   ./run_openmp_scaling.sh <path-to-binary> <N> <reps> <output-csv>
#
# Example (matmul, what we're running today):
#   ./run_openmp_scaling.sh ./bin/matmul_omp 1024 5 results/openmp_scaling.csv
#
# Example (reduction, later):
#   ./run_openmp_scaling.sh ./bin/reduction_omp 8388608 5 results/openmp_scaling.csv
#
# Thread counts are fixed below, matching what we agreed for matmul:
# 1, 2, 4, 6 (one thread per physical core), 8, 12 (all logical threads).
# Edit THREADS= if a future kernel needs a different set.

set -e  # stop immediately if any command fails, rather than continuing
        # with a partially-broken run

BINARY="$1"
N="$2"
REPS="$3"
OUTPUT_CSV="$4"

if [ -z "$BINARY" ] || [ -z "$N" ] || [ -z "$REPS" ] || [ -z "$OUTPUT_CSV" ]; then
    echo "Usage: $0 <path-to-binary> <N> <reps> <output-csv>"
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "Error: $BINARY does not exist or is not executable."
    exit 1
fi

THREADS=(1 2 4 6 8 12)

echo "Running $BINARY at N=$N, reps=$REPS, threads={${THREADS[*]}}"
echo "Appending CSV rows to $OUTPUT_CSV"
echo

for t in "${THREADS[@]}"; do
    echo "--- threads=$t ---"
    # stdout (the CSV line) is appended to the results file.
    # stderr (the "sanity: ..." line) still prints to the terminal
    # so you can eyeball correctness on every single run, same as
    # we've been doing by hand so far.
    "$BINARY" -n "$N" -r "$REPS" -t "$t" >> "$OUTPUT_CSV"
    echo
done

echo "Done. Current contents of $OUTPUT_CSV:"
cat "$OUTPUT_CSV"
