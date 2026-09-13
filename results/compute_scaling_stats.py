#!/usr/bin/env python3
"""
results/compute_scaling_stats.py

Reads the three raw, machine-generated CSVs produced by the run-scripts
this session (results/openmp_scaling.csv, results/mpi_scaling.csv,
results/cuda_bench.csv) and produces three SEPARATE summary CSVs:
  - results/openmp_scaling_summary.csv
  - results/mpi_scaling_summary.csv
  - results/cuda_bench_summary.csv

This script only AVERAGES and computes derived numbers (speedup,
efficiency, %-of-peak-bandwidth). It does not re-run anything, and it
does not plot anything -- that's results/plot.py, reading these
summary files, written separately.

Beginner note on why three separate output files instead of one: the
raw CSVs all share the same nine columns (kernel, paradigm, precision,
N, threads_or_ranks_or_blockdim, time_min, time_med, gflops, gbytes_s),
but that fifth column means something different in each paradigm --
OpenMP thread count and MPI rank count are both "how many workers,
speedup should ideally scale with this", while CUDA's threads-per-
block/tile-dim/block-dim is a fixed hardware-tuning constant, not a
worker count to compute speedup against. Forcing all three into one
schema with one "speedup" column would misrepresent what CUDA's number
means, so they get their own summary files and their own logic
instead.

Usage:
    python3 compute_scaling_stats.py
(run from the repo root -- it expects results/*.csv to exist relative
to the current directory, same assumption the shell scripts made)
"""

import csv
import statistics
from collections import defaultdict

# The measured Device-to-Device bandwidth ceiling from bandwidthTest,
# re-run and reconfirmed this session (82.66 GB/s -- NOT the original
# Phase 0 figure of ~72.4 GB/s, which bandwidthTest's own output
# warned can vary run-to-run under GPU Boost). This is a plain
# constant, not something the script can look up itself -- if you
# re-run bandwidthTest again later and get a different number, update
# this by hand.
PEAK_BANDWIDTH_GBPS = 82.66

# Column layout shared by every raw CSV row, per common/bench.c's
# bench_report_csv format.
COLUMNS = ["kernel", "paradigm", "precision", "N", "workers",
           "time_min", "time_med", "gflops", "gbytes_s"]


def read_raw_csv(path):
    """Read a raw bench CSV into a list of dicts, one per row.
    Numeric columns are converted from string to float/int here so
    every later step can just do arithmetic without re-parsing."""
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) != len(COLUMNS):
                    # Skip malformed lines (e.g. a stray blank line)
                    # rather than crashing the whole script on one
                    # bad row.
                    continue
                row = dict(zip(COLUMNS, line))
                row["N"] = int(row["N"])
                row["workers"] = int(row["workers"])
                row["time_min"] = float(row["time_min"])
                row["time_med"] = float(row["time_med"])
                row["gflops"] = float(row["gflops"])
                row["gbytes_s"] = float(row["gbytes_s"])
                rows.append(row)
    except FileNotFoundError:
        print(f"Note: {path} not found, skipping.")
    return rows


def average_by_group(rows, group_keys):
    """Group rows by a tuple of column names (e.g. kernel/precision/N/
    workers) and average time_med and gbytes_s within each group. This
    is the "average across repeated runs" step -- e.g. matmul's three
    separate run_openmp_scaling.sh invocations at threads=6 all land
    in the same group here and get collapsed into one averaged row."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row)

    averaged = []
    for key, group_rows in groups.items():
        entry = dict(zip(group_keys, key))
        entry["num_samples"] = len(group_rows)
        entry["avg_time_med"] = statistics.mean(r["time_med"] for r in group_rows)
        entry["avg_gbytes_s"] = statistics.mean(r["gbytes_s"] for r in group_rows)
        entry["avg_gflops"] = statistics.mean(r["gflops"] for r in group_rows)
        averaged.append(entry)
    return averaged


def compute_speedup_efficiency(averaged_rows):
    """For OpenMP/MPI-style data: within each (kernel, precision, N)
    group, find the workers=1 baseline time and compute speedup/
    efficiency for every worker count relative to it. Kernels with no
    workers=1 row present (shouldn't happen given how we ran things,
    but defensive) are skipped with a note rather than crashing."""
    by_kernel = defaultdict(list)
    for row in averaged_rows:
        by_kernel[(row["kernel"], row["precision"], row["N"])].append(row)

    results = []
    for (kernel, precision, n), rows in by_kernel.items():
        baseline_rows = [r for r in rows if r["workers"] == 1]
        if not baseline_rows:
            print(f"Note: no workers=1 baseline for {kernel}/{precision}/N={n}, "
                  f"skipping speedup calc for this group.")
            continue
        baseline_time = baseline_rows[0]["avg_time_med"]

        for row in sorted(rows, key=lambda r: r["workers"]):
            speedup = baseline_time / row["avg_time_med"]
            efficiency = speedup / row["workers"]
            results.append({
                "kernel": kernel,
                "precision": precision,
                "N": n,
                "workers": row["workers"],
                "num_samples": row["num_samples"],
                "avg_time_med": row["avg_time_med"],
                "speedup": speedup,
                "efficiency_pct": efficiency * 100.0,
            })
    return results


def compute_cuda_summary(averaged_rows):
    """For CUDA: no speedup to compute (there's no baseline "1 worker"
    concept the way OpenMP/MPI have it) -- instead, report achieved
    bandwidth as a percentage of the measured peak."""
    results = []
    for row in averaged_rows:
        results.append({
            "kernel": row["kernel"],
            "paradigm": row["paradigm"],
            "precision": row["precision"],
            "N": row["N"],
            "block_param": row["workers"],
            "num_samples": row["num_samples"],
            "avg_gbytes_s": row["avg_gbytes_s"],
            "avg_gflops": row["avg_gflops"],
            "pct_of_peak_bandwidth": (row["avg_gbytes_s"] / PEAK_BANDWIDTH_GBPS) * 100.0,
        })
    return results


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    # --- OpenMP ---
    omp_rows = read_raw_csv("results/openmp_scaling.csv")
    if omp_rows:
        omp_avg = average_by_group(omp_rows, ["kernel", "precision", "N", "workers"])
        omp_summary = compute_speedup_efficiency(omp_avg)
        write_csv(
            "results/openmp_scaling_summary.csv",
            omp_summary,
            ["kernel", "precision", "N", "workers", "num_samples",
             "avg_time_med", "speedup", "efficiency_pct"],
        )

    # --- MPI ---
    mpi_rows = read_raw_csv("results/mpi_scaling.csv")
    if mpi_rows:
        mpi_avg = average_by_group(mpi_rows, ["kernel", "precision", "N", "workers"])
        mpi_summary = compute_speedup_efficiency(mpi_avg)
        write_csv(
            "results/mpi_scaling_summary.csv",
            mpi_summary,
            ["kernel", "precision", "N", "workers", "num_samples",
             "avg_time_med", "speedup", "efficiency_pct"],
        )

    # --- CUDA ---
    # NOTE: paradigm MUST be included in the grouping keys here, unlike
    # OpenMP/MPI. For those two, the "paradigm" column never varies
    # within a kernel (always "omp" or "mpi"), so grouping by kernel
    # alone works. For CUDA, the *kernel* column only says "reduction"
    # or "transpose" -- the actual variant (atomic/shared/shuffle,
    # naive/tiled) is carried in the paradigm column instead
    # (cuda_atomic, cuda_shared, cuda_naive, cuda_tiled, etc). Grouping
    # by kernel alone silently merges atomic+shared+shuffle into one
    # averaged row -- caught by testing this script against real data
    # before handing it over, not something to assume works.
    cuda_rows = read_raw_csv("results/cuda_bench.csv")
    if cuda_rows:
        cuda_avg = average_by_group(cuda_rows, ["kernel", "paradigm", "precision", "N", "workers"])
        cuda_summary = compute_cuda_summary(cuda_avg)
        write_csv(
            "results/cuda_bench_summary.csv",
            cuda_summary,
            ["kernel", "paradigm", "precision", "N", "block_param", "num_samples",
             "avg_gbytes_s", "avg_gflops", "pct_of_peak_bandwidth"],
        )


if __name__ == "__main__":
    main()
