# Matmul

**The algorithm.** Standard dense matrix multiply, `C = A × B` for square N×N matrices — same O(N³) work as the textbook triple loop, but with loop order `i → k → j` instead of the naive `i → j → k`. That reordering hoists `A[i][k]` to a scalar once per `(i,k)` pair, then walks both `B[k][j]` and `C[i][j]` along a contiguous row in the innermost loop — much better cache-line reuse than the naive ordering's strided column-walk through `B`. Same result, same op count, purely a memory-access-pattern change.

**The decomposition strategy.**
- **`serial.c`** — the `ikj` triple loop above, single thread. Baseline.
- **`omp.c`** — `#pragma omp parallel for schedule(static)` splits the *outer* `i` loop across threads: each thread owns a contiguous block of output rows, computed independently, no communication needed mid-compute.
- **`mpi.c` (Cannon's algorithm)** — a real change in decomposition, not just "OpenMP but processes." Ranks are arranged on a **q × q grid** (`q = √p`), each starting with one `nb × nb` block of `A`, `B`, and `C` (`nb = N/q`). Blocks are pre-skewed once at the start (row *i* of `A` shifted left by *i*, column *j* of `B` shifted up by *j*, both wrapping around) so that each rank's local block-pair is already correctly aligned for multiplication. Then `q` rounds run: local block-multiply-accumulate, followed by a uniform one-step wraparound shift (`A` left, `B` up) — after `q` rounds every rank has multiplied against every block it needed to, with only nearest-neighbor communication, never a full broadcast. **This requires `p` to be a perfect square and `N` divisible by `√p`, both enforced with explicit validation and an error message.**

**What was measured.**

*OpenMP thread scaling, N=1024 (speedup / efficiency):*
| Threads | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| Speedup | 1.00× | 1.42× | 2.19× | 2.93× | 2.74× | 3.40× |
| Efficiency | 100% | 71.06% | 54.68% | 48.76% | 34.30% | 28.30% |

*MPI rank scaling, N=6912 (only perfect-square rank counts are valid for Cannon's grid):*
| Ranks | 1 | 4 | 9 |
|---|---|---|---|
| Speedup | 1.00× | 1.50× | 1.43× |
| Efficiency | 100% | 37.45% | 15.92% |

OpenMP and MPI were benchmarked at different N (1024 vs. 6912): MPI needs a larger problem to amortize Cannon's per-round shift/communication cost, while OpenMP's smaller N kept the single-threaded baseline fast enough to iterate on during development.

**What was surprising.**
- The same hyperthreading crossover shows up here as in reduction, just on compute-bound work instead of bandwidth-bound: efficiency drops from 48.76% (6 threads) to 34.30% (8 threads) — this machine only has 6 physical cores, so threads past 6 share one instead of getting their own.
- **9 ranks is an outright regression, not just a lower-efficiency plateau.** Wall-clock time at 9 ranks (39.7s) is worse than at 4 ranks (38.0s), despite running on more than double the ranks. With only 6 physical cores available, 9 MPI ranks means some ranks are competing for the same core (oversubscription) on top of Cannon's per-round communication cost — the two overheads stack instead of one hiding the other.
