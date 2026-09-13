#!/bin/bash
#
# results/run_cuda_bench.sh
#
# Runs a LIST of CUDA kernel binaries (each with its own N and
# threads_per_block -- unlike OpenMP/MPI, CUDA kernels aren't a single
# binary swept across a range, they're several different programs, each
# already sized appropriately for its own memory footprint). Each entry
# is run NUM_RUNS times for averaging, same discipline as the OpenMP
# and MPI scripts. Does NOT compute bandwidth %-of-peak or plot
# anything -- that's compute_scaling_stats.py and plot.py, run later
# once all of Phase 5's data (OpenMP + MPI + CUDA) is collected.
#
# Why N and threads_per_block are per-entry, not one shared value for
# all kernels (unlike run_openmp_scaling.sh's single N): double
# precision uses 8 bytes/element instead of float's 4, so a double-
# precision run at the SAME N as its float counterpart would use ~2x
# the GPU memory -- and this machine only has ~1 GB free VRAM (Phase
# 0's measurement). Each entry gets its own N specifically so a smaller
# N can be chosen for double-precision entries without changing the
# script itself.
#
# Usage:
#   ./run_cuda_bench.sh <output-csv> <num-runs> "<entry1>;<entry2>;..."
#
# Each entry is "binary,N,threads_or_tile_or_block_dim,reps" (comma-
# separated fields, semicolon-separated entries -- semicolons need the
# whole list quoted as one argument, same reasoning as
# run_mpi_scaling.sh's ranks string).
#
# reps is per-entry, not one shared value, because it doesn't mean the
# same thing for every binary: for saxpy/reduction/transpose it's
# "how many timed repetitions bench.c runs for min/median stats" (5 is
# typical). For jacobi_cuda specifically, the binary's own -r flag
# means "how many stencil iterations to run" -- an algorithmic
# parameter, NOT a measurement-repetition count -- so its entry needs
# a much larger number (e.g. 1000) for the timing to reflect steady-
# state behavior rather than a handful of startup sweeps.
#
# Example (today's four already-seen kernels, captured to disk this time):
#   ./run_cuda_bench.sh results/cuda_bench.csv 3 \
#     "./bin/saxpy_cuda,50000000,256,5;./bin/reduce_atomic,50000000,256,5;./bin/reduce_shared,50000000,256,5;./bin/reduce_shuffle,50000000,256,5"

set -e  # stop immediately if any command fails

OUTPUT_CSV="$1"
NUM_RUNS="$2"
ENTRIES_STR="$3"

if [ -z "$OUTPUT_CSV" ] || [ -z "$NUM_RUNS" ] || [ -z "$ENTRIES_STR" ]; then
    echo "Usage: $0 <output-csv> <num-runs> \"<binary,N,threads>;...\""
    exit 1
fi

# split the semicolon-separated entry list into an array
IFS=';' read -ra ENTRIES <<< "$ENTRIES_STR"

echo "Running ${#ENTRIES[@]} CUDA entries, repeated $NUM_RUNS time(s)"
echo "Appending CSV rows to $OUTPUT_CSV"
echo

for run in $(seq 1 "$NUM_RUNS"); do
    echo "=== run $run of $NUM_RUNS ==="
    for entry in "${ENTRIES[@]}"; do
        # split "binary,N,threads,reps" on commas
        IFS=',' read -r BINARY N THREADS REPS <<< "$entry"

        if [ ! -x "$BINARY" ]; then
            echo "Error: $BINARY does not exist or is not executable. Skipping."
            continue
        fi

        echo "--- $BINARY -n $N -t $THREADS -r $REPS ---"
        # stdout (the CSV line) is appended to the results file.
        # stderr (the "sanity: ..." line) still prints to the terminal
        # so correctness can be eyeballed on every single run, same
        # discipline as the OpenMP and MPI scripts.
        "$BINARY" -n "$N" -r "$REPS" -t "$THREADS" >> "$OUTPUT_CSV"
        echo
    done
done

echo "Done. Current contents of $OUTPUT_CSV:"
cat "$OUTPUT_CSV"
