#!/bin/bash
#
# results/run_mpi_scaling.sh
#
# Runs one MPI kernel binary across a fixed set of rank counts, repeated
# NUM_RUNS times for averaging, appending each run's CSV line to a
# results file. Does NOT plot or average anything -- that's done later,
# by hand or in results/plot.py, once the CSV has real data in it.
#
# Unlike OpenMP (where thread count is a -t flag read by the program
# itself), MPI programs are LAUNCHED by a separate command -- mpirun --
# which starts <ranks> separate copies of the binary as independent OS
# processes ("ranks"), each with its own memory. So parallelism here
# comes from how we invoke mpirun, not from a flag the binary parses.
#
# --oversubscribe: this machine has 6 physical cores. Asking for 9
# ranks means OpenMPI is told to place more processes than it thinks it
# has "slots" for, and it will often refuse outright without this flag.
# It does NOT mean 9 ranks will run as fast as 6 -- it just means
# OpenMPI is allowed to place more than one rank per core if needed.
#
# Usage:
#   ./run_mpi_scaling.sh <path-to-binary> <N> <reps> <num-runs> <output-csv> "<ranks>"
#
# Example (matmul -- perfect squares only, per its Cannon's-algorithm
# decomposition: p must be a perfect square, N divisible by sqrt(p)):
#   ./run_mpi_scaling.sh ./bin/matmul_mpi 6912 5 3 results/mpi_scaling.csv "1 4 9"
#
# Example (matvec/gaussian/reduction -- no perfect-square constraint,
# mirroring the OpenMP thread set for a direct ranks-vs-threads comparison):
#   ./run_mpi_scaling.sh ./bin/matvec_mpi 6144 5 3 results/mpi_scaling.csv "1 2 4 6 8"
#
# RANKS is passed in as a single quoted, space-separated string (the
# quotes matter -- without them the shell would treat each number as a
# separate argument instead of one list). Different MPI kernels have
# different constraints on valid rank counts, so this is supplied per
# call rather than hardcoded.

set -e  # stop immediately if any command fails

BINARY="$1"
N="$2"
REPS="$3"
NUM_RUNS="$4"
OUTPUT_CSV="$5"
RANKS_STR="$6"

if [ -z "$BINARY" ] || [ -z "$N" ] || [ -z "$REPS" ] || [ -z "$NUM_RUNS" ] || [ -z "$OUTPUT_CSV" ] || [ -z "$RANKS_STR" ]; then
    echo "Usage: $0 <path-to-binary> <N> <reps> <num-runs> <output-csv> \"<ranks>\""
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "Error: $BINARY does not exist or is not executable."
    exit 1
fi

# read the quoted ranks string into an array, e.g. "1 2 4 6 8" -> (1 2 4 6 8)
read -ra RANKS <<< "$RANKS_STR"

echo "Running $BINARY at N=$N, reps=$REPS, ranks={${RANKS[*]}}, repeated $NUM_RUNS time(s)"
echo "Appending CSV rows to $OUTPUT_CSV"
echo

for run in $(seq 1 "$NUM_RUNS"); do
    echo "=== run $run of $NUM_RUNS ==="
    for p in "${RANKS[@]}"; do
        echo "--- ranks=$p ---"
        # stdout (the CSV line) is appended to the results file.
        # stderr (any sanity-check output the binary prints) still
        # prints to the terminal so correctness can be eyeballed on
        # every single run, same discipline as the OpenMP script.
        mpirun --oversubscribe -np "$p" "$BINARY" -n "$N" -r "$REPS" >> "$OUTPUT_CSV"
        echo
    done
done

echo "Done. Current contents of $OUTPUT_CSV:"
cat "$OUTPUT_CSV"
