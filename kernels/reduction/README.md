# Reduction

**The algorithm.** Reduction collapses an array of N values down to one number via repeated combination (here, addition) — the classic "map, then reduce" pattern. Two modes share one code path: `sum` reduces an array of random values (verified against `reference/check_reduction.py`), and `pi` estimates π via the trapezoidal rule — sampling N midpoints of a quarter-circle function and summing slice areas, so numeric integration and plain summation turn out to be the same operation with a different "map" step first. `pi` converges as O(1/N), so its tolerance is deliberately loose (1e-2) rather than tight.

**The decomposition strategy.** Six variants split the same reduction six different ways:
- **`serial.c`** — one loop, one accumulator. The correctness/timing baseline everything else is checked against.
- **`omp.c`** — `#pragma omp parallel for reduction(+:acc)`: OpenMP statically splits the array across threads, each keeps a private partial sum, and the compiler generates the final combine step automatically.
- **`mpi.c`** — the array is split by rank, each rank sums its own slice, then `MPI_Reduce` combines all ranks' partial sums (a tree-structured combine across separate processes, not shared memory).
- **`mpi_subcomm.c`** — same idea, but reduces across a *sub*-communicator: a named subset of ranks, rather than every rank in the job. Useful when only part of a larger run needs to participate in a given reduction.
- **`mpi_omp.c`** — hybrid, two decomposition levels at once: MPI splits the array across ranks (processes, possibly different machines), then each rank further splits its own slice across OpenMP threads.
- **CUDA — three variants, naive to optimized:** `reduce_atomic` (every GPU thread does a global `atomicAdd` — simplest, heaviest contention), `reduce_shared` (threads combine within a block via shared memory first, one `atomicAdd` per block instead of per thread), `reduce_shuffle` (the final 32 threads combine via `__shfl_down_sync`, register-to-register, skipping memory entirely).

**What was measured.**

*OpenMP thread scaling (efficiency):*
| Threads | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| Efficiency | 100% | 95.93% | 89.18% | 87.31% | 65.14% | 63.89% |

*MPI rank scaling (efficiency):*
| Ranks | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Efficiency | 100% | 64.03% | 42.33% | 27.98% | 20.26% |

*CUDA, float, achieved bandwidth (% of measured peak, 82.66 GB/s):*
| Variant | GB/s | % of peak |
|---|---|---|
| `reduce_atomic` | 1.98 | 2.4% |
| `reduce_shared` | 46.08 | 55.7% |
| `reduce_shuffle` | 72.75 | 88.0% |

*CUDA, double:* `reduce_atomic` 3.98 GB/s (4.8%), `reduce_shared` 60.42 GB/s (73.1%), `reduce_shuffle` 85.97 GB/s (104.0% — see note below).

**What was surprising.**
- OpenMP's efficiency holds above 85% through 6 threads, then drops sharply at 8 — matching this machine's actual topology (6 physical cores, each with 2 hyperthreads, confirmed via `lstopo`). Threads past 6 end up sharing a core instead of getting one to themselves.
- MPI degrades much faster than OpenMP at the same worker counts (64% at 2 ranks vs. 96% at 2 threads). Some of that gap is expected — MPI's inter-process communication has no equivalent on OpenMP's shared memory — but the steepness of it still stood out.
- The CUDA atomic→shuffle progression hit **~36.7× bandwidth improvement** (float), well past the ~10× I was expecting going in. `reduce_atomic`'s contention turned out worse in practice than the naive estimate suggested.
- `reduce_shuffle` (double) measured 85.97 GB/s — **104.0%** of the measured `bandwidthTest` peak (82.66 GB/s). Not a measurement error: `bandwidthTest` itself varies run-to-run under GPU Boost clocks (its own output warns of this), so a "peak" figure from one run can be beaten by a different kernel caught at a slightly higher boost state. Reported as measured, not artificially capped at 100%.
