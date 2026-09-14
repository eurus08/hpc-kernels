# Transpose

**The algorithm.** Matrix transpose, `B[j][i] = A[i][j]` — pure data rearrangement, zero arithmetic. That makes it **bandwidth-bound**: the bottleneck is moving bytes, not computing anything, which shows up clearly in the results below.

**The decomposition strategy.**
- **`serial.c`** — baseline nested loop.
- **`omp.c`** — the outer loop parallelized via `#pragma omp parallel for`, deliberately plain (no CPU-side cache blocking/tiling) — kept simple since, as the results below show, thread count barely moves the needle on a bandwidth-bound kernel anyway.
- **`transpose_naive.cu`** — each GPU thread reads one element and writes it to its transposed position directly in global memory. Transpose has an unavoidable structural problem here: whichever direction (read or write) walks against the matrix's row-major layout ends up strided across memory instead of contiguous — you cannot make both directions coalesced simultaneously without an intermediate step.
- **`transpose_tiled.cu`** — the intermediate step: each thread block first stages a `TILE × TILE` chunk of `A` into on-chip shared memory via a **fully coalesced** read, then writes that chunk back out to `B` via a **fully coalesced** write — the transposition itself happens inside fast shared memory, where the strided-access cost is far cheaper, instead of happening directly against slow global memory. The shared-memory tile is declared with a **"+1" padding column** specifically to avoid bank conflicts during the transposed read-back — and since tile size comes from the runtime `-t` flag rather than being a compile-time constant, this uses dynamically-sized shared memory with manual row-stride indexing instead of a simple static 2D array.

**What was measured.**

*OpenMP thread scaling, N=4096 (speedup / efficiency):*
| Threads | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| Speedup | 1.00× | 1.48× | 1.53× | 1.44× | 1.53× | 1.56× |
| Efficiency | 100% | 73.87% | 38.34% | 24.01% | 19.06% | 13.03% |

*CUDA, naive vs. tiled, achieved bandwidth (% of peak):*
| Precision | N | Naive | Tiled |
|---|---|---|---|
| float | 8192 | 43.47% | 99.07% |
| double | 5760 | 76.19% | 99.87% |

**What was surprising.**
- OpenMP speedup essentially caps at **~1.5×** and stays flat from 2 threads all the way to 12 — a clean, direct confirmation that transpose is bandwidth-bound: extra CPU cores can't help once everyone is competing for the same memory bus, in sharp contrast to matmul's genuine 3.4× speedup at 12 threads on the *same* hardware for the *same* thread counts. It's the clearest compute-bound-vs-bandwidth-bound contrast anywhere in this repo.
- The naive→tiled CUDA jump (43.5%→99.1% float, 76.2%→99.9% double) makes coalescing visible: the entire gap traces to converting non-coalesced strided memory transactions into coalesced ones by routing through shared memory first — nothing else about the algorithm changed.
