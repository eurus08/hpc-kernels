#!/usr/bin/env python3
"""
results/plot.py

Reads the three SUMMARY CSVs (produced by compute_scaling_stats.py --
NOT the raw per-run CSVs) and produces three PNGs in results/:
  - openmp_speedup.png
  - mpi_strong_scaling.png
  - roofline.png

This script only reads already-averaged data and draws pictures. It
does not run any binaries, and it does not compute speedup/efficiency
itself -- that's compute_scaling_stats.py's job, run first.

Usage:
    python3 plot.py
(run from the repo root -- expects results/*_summary.csv to exist)
"""

import csv
import matplotlib.pyplot as plt

# Peak figures, confirmed via deviceQuery on this machine (Quadro P2000
# Max-Q, Pascal, 768 CUDA cores @ 1468 MHz max clock) and bandwidthTest
# (re-run and reconfirmed later on, superseding an earlier ~72.4 GB/s
# measurement from the first pass).
PEAK_BANDWIDTH_GBPS = 82.66
PEAK_GFLOPS_FP32 = 2 * 768 * 1.468       # 2 FLOPs/cycle (FMA) x cores x GHz
PEAK_GFLOPS_FP64 = PEAK_GFLOPS_FP32 / 32  # Pascal non-flagship FP64 = 1/32 of FP32


def read_csv_dicts(path):
    """Read a summary CSV into a list of dicts, converting numeric
    fields from string to float/int as needed."""
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"Note: {path} not found, skipping the plot(s) that need it.")
    return rows


def plot_scaling(rows, worker_label, title, out_path):
    """Shared logic for the OpenMP and MPI plots: one line per kernel,
    speedup vs. worker count, plus a dashed ideal-scaling reference
    line (y = x, i.e. perfect linear speedup)."""
    if not rows:
        return

    kernels = sorted(set(r["kernel"] for r in rows))
    max_workers = max(int(r["workers"]) for r in rows)

    fig, ax = plt.subplots(figsize=(8, 6))

    for kernel in kernels:
        kernel_rows = sorted(
            (r for r in rows if r["kernel"] == kernel),
            key=lambda r: int(r["workers"]),
        )
        workers = [int(r["workers"]) for r in kernel_rows]
        speedups = [float(r["speedup"]) for r in kernel_rows]
        ax.plot(workers, speedups, marker="o", label=kernel)

    # Ideal-scaling reference line: speedup == worker count exactly.
    ideal_x = [1, max_workers]
    ax.plot(ideal_x, ideal_x, linestyle="--", color="gray", label="ideal (linear)")

    ax.set_xlabel(worker_label)
    ax.set_ylabel("Speedup vs. 1 worker")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_roofline(rows, out_path):
    """One log-log chart: a diagonal memory-bandwidth roof (slope =
    peak GB/s), two flat compute roofs (FP32 and FP64 peak GFLOP/s),
    and every CUDA kernel plotted as a point at (arithmetic intensity,
    achieved GFLOP/s).

    Arithmetic intensity = FLOPs per byte moved = avg_gflops /
    avg_gbytes_s (both already in "per second" units, so the seconds
    cancel out, leaving FLOPs/byte). Transpose kernels report zero
    measured GFLOP/s (pure data movement, no arithmetic) -- true
    arithmetic intensity there is exactly zero, not just very small,
    so they're floored to a small plotting-only value (MIN_AI) purely
    so log-scale axes can display them; the near-zero position is
    itself the correct story for a kernel that does no compute.
    """
    if not rows:
        return

    MIN_AI = 1e-3  # plotting floor only, not a measured value

    fig, ax = plt.subplots(figsize=(11, 8))

    # Two roof colors, since float and double kernels share the same
    # bandwidth roof but hit different compute-roof ceilings.
    colors = {"float": "tab:blue", "double": "tab:orange"}

    # Staggered label offsets, cycled by point index -- with 14 points
    # clustered in a narrow arithmetic-intensity band (typical for
    # bandwidth-bound kernels), a single fixed offset makes every label
    # overlap. Cycling through several offsets spreads them out enough
    # to stay readable.
    offset_cycle = [(8, 8), (8, -14), (-90, 8), (-90, -14), (8, 20), (-90, 22)]

    for idx, row in enumerate(rows):
        gflops = float(row["avg_gflops"])
        gbytes_s = float(row["avg_gbytes_s"])
        ai = (gflops / gbytes_s) if gbytes_s > 0 else 0.0
        ai_plot = max(ai, MIN_AI)

        precision = row["precision"]
        color = colors.get(precision, "black")
        label = f"{row['kernel']}/{row['paradigm']} ({precision})"

        y_plot = max(gflops, 1e-3)
        ax.scatter(ai_plot, y_plot, color=color, zorder=5, s=40)
        dx, dy = offset_cycle[idx % len(offset_cycle)]
        ax.annotate(label, (ai_plot, y_plot),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # Roofline geometry: x-axis is arithmetic intensity (FLOPs/byte),
    # y-axis is achieved GFLOP/s. The memory roof is a diagonal line
    # (GFLOP/s = intensity x bandwidth) up until it crosses the flat
    # compute roof (GFLOP/s = peak compute) -- that crossing point is
    # the "ridge point."
    x_range = [MIN_AI, 1000]

    ridge_fp32 = PEAK_GFLOPS_FP32 / PEAK_BANDWIDTH_GBPS
    mem_roof_fp32 = [MIN_AI * PEAK_BANDWIDTH_GBPS, ridge_fp32 * PEAK_BANDWIDTH_GBPS]
    ax.plot([MIN_AI, ridge_fp32], mem_roof_fp32, color="gray", linestyle="-")
    ax.plot([ridge_fp32, x_range[1]], [PEAK_GFLOPS_FP32, PEAK_GFLOPS_FP32],
            color="tab:blue", linestyle="-", label=f"FP32 peak ({PEAK_GFLOPS_FP32:.0f} GFLOP/s)")

    ridge_fp64 = PEAK_GFLOPS_FP64 / PEAK_BANDWIDTH_GBPS
    mem_roof_fp64 = [MIN_AI * PEAK_BANDWIDTH_GBPS, ridge_fp64 * PEAK_BANDWIDTH_GBPS]
    ax.plot([MIN_AI, ridge_fp64], mem_roof_fp64, color="gray", linestyle="--")
    ax.plot([ridge_fp64, x_range[1]], [PEAK_GFLOPS_FP64, PEAK_GFLOPS_FP64],
            color="tab:orange", linestyle="-", label=f"FP64 peak ({PEAK_GFLOPS_FP64:.1f} GFLOP/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved performance (GFLOP/s)")
    ax.set_title(f"Roofline -- Quadro P2000 Max-Q (peak BW {PEAK_BANDWIDTH_GBPS} GB/s)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    omp_rows = read_csv_dicts("results/openmp_scaling_summary.csv")
    plot_scaling(omp_rows, "OpenMP threads", "OpenMP Speedup", "results/openmp_speedup.png")

    mpi_rows = read_csv_dicts("results/mpi_scaling_summary.csv")
    plot_scaling(mpi_rows, "MPI ranks", "MPI Strong Scaling", "results/mpi_strong_scaling.png")

    cuda_rows = read_csv_dicts("results/cuda_bench_summary.csv")
    plot_roofline(cuda_rows, "results/roofline.png")


if __name__ == "__main__":
    main()
