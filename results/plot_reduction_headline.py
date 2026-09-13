#!/usr/bin/env python3
"""
plot_reduction_headline.py

Builds the single headline plot for the root README: reduction's
efficiency curve under OpenMP vs. under MPI, on one chart.

Reads:
    results/openmp_scaling_summary.csv
    results/mpi_scaling_summary.csv

Writes:
    results/reduction_headline.png

Only stdlib csv + matplotlib — no numpy/scipy needed, this is pure
plotting of already-aggregated numbers.
"""

import csv
import pathlib
import matplotlib.pyplot as plt

# Directory this script lives in, regardless of what directory it's run
# from (fixes FileNotFoundError when invoked as e.g. `python3 results/plot_reduction_headline.py`
# from the repo root instead of from inside results/).
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_reduction_rows(path):
    """Read a summary CSV and return only the rows where kernel == reduction,
    sorted by worker count. Each row becomes a dict."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["kernel"] == "reduction":
                rows.append(row)
    rows.sort(key=lambda r: int(r["workers"]))
    return rows


def main():
    omp_rows = load_reduction_rows(SCRIPT_DIR / "openmp_scaling_summary.csv")
    mpi_rows = load_reduction_rows(SCRIPT_DIR / "mpi_scaling_summary.csv")

    if not omp_rows:
        raise SystemExit("No reduction rows found in openmp_scaling_summary.csv")
    if not mpi_rows:
        raise SystemExit("No reduction rows found in mpi_scaling_summary.csv")

    omp_workers = [int(r["workers"]) for r in omp_rows]
    omp_eff = [float(r["efficiency_pct"]) for r in omp_rows]

    mpi_workers = [int(r["workers"]) for r in mpi_rows]
    mpi_eff = [float(r["efficiency_pct"]) for r in mpi_rows]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(omp_workers, omp_eff, marker="o", linewidth=2,
            label="OpenMP (threads)", color="#1f77b4")
    ax.plot(mpi_workers, mpi_eff, marker="s", linewidth=2,
            label="MPI (ranks)", color="#d62728")

    ax.axhline(100, color="gray", linestyle=":", linewidth=1,
                label="Ideal (100%)")

    ax.set_xlabel("Number of workers (threads or ranks)")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("Reduction: OpenMP vs. MPI scaling efficiency\n(same algorithm, same problem class, two paradigms)")
    ax.set_ylim(0, 110)
    ax.set_xticks(sorted(set(omp_workers) | set(mpi_workers)))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = SCRIPT_DIR / "reduction_headline.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # Print the numbers actually plotted, so they can be sanity-checked
    # against the summary CSVs by eye before trusting the image.
    print("\nOpenMP reduction (workers, efficiency%):")
    for w, e in zip(omp_workers, omp_eff):
        print(f"  {w:>3}  {e:6.2f}%")
    print("\nMPI reduction (workers, efficiency%):")
    for w, e in zip(mpi_workers, mpi_eff):
        print(f"  {w:>3}  {e:6.2f}%")


if __name__ == "__main__":
    main()
