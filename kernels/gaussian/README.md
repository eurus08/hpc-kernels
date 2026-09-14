# Gaussian Elimination

**The algorithm.** Solves `Ax = b` via forward elimination (reduce `A` to upper-triangular form) followed by back-substitution. As documented in the root README's provenance section, this implementation deliberately **skips partial pivoting** — input matrices are generated diagonally dominant, which guarantees numerical stability without needing to search for and swap the largest-magnitude pivot row each step.

**The decomposition strategy — cyclic, not block, and the only kernel in this repo decomposed that way.** Row `i` is owned by rank `(i mod p)` — rank 0 owns rows 0, p, 2p...; rank 1 owns rows 1, p+1, 2p+1...; and so on. This is a deliberate choice, not an arbitrary one: elimination doesn't touch every row equally over time. By pivot step k, rows 0..k are already finished; only rows k+1..n-1 still have real work. Under a *block* distribution (like matvec's), whichever rank owns the earliest rows runs out of useful work as k grows and sits idle for the rest of the run. Under *cyclic* distribution, every rank still owns roughly `(n-k)/p` of the remaining unprocessed rows at every step, regardless of how far elimination has progressed — the standard textbook fix for load-balancing elimination-shaped algorithms specifically.

This has a real communication cost, though: because cyclic rows aren't contiguous in memory (unlike matvec's block rows), the one-time initial distribution can't use a single `MPI_Scatterv` — rank 0 loops over all rows and sends each individually to its owner. More importantly, **every pivot step requires a fresh broadcast**: the rank owning row `k` packs that row plus `b[k]` and `MPI_Bcast`s it to everyone, so every other rank can eliminate it from their own remaining rows. Back-substitution runs the same pattern in reverse, one broadcast per unknown solved. That's **O(N) total broadcasts** — one per pivot step, one per back-substitution step — a fundamentally more communication-heavy pattern than matvec's one-time scatter or Cannon's O(√p) shift rounds, because it's forced by the algorithm's inherently sequential row-dependency structure, not a decomposition choice.

**What was measured.**

*MPI rank scaling, N=1536 (speedup / efficiency):*
| Ranks | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Speedup | 1.00× | 1.76× | 2.66× | 2.86× | 2.65× |
| Efficiency | 100% | 88.04% | 66.52% | 47.63% | 33.12% |

**What was surprising.**
- Efficiency drops off faster than any MPI kernel here except reduction — a direct consequence of the O(N) broadcast count above: communication frequency doesn't shrink as rank count grows, unlike matvec's single scatter or Cannon's O(√p) rounds.
- **8 ranks is an outright regression vs. 6 ranks** (2.65× vs. 2.86× speedup) — the same core-oversubscription story seen independently in matmul's 9-rank result: past 6 physical cores, extra ranks compete for the same core on top of the algorithm's own communication cost.
- **`sum(x)` came back bit-for-bit identical across every rank count tested (1, 2, 4, 6, 8)** — the only MPI kernel in this repo where that's true. Every other MPI kernel (matmul, matvec, reduction) showed the expected floating-point summation-order drift between different decompositions. Gaussian elimination's strictly sequential row-dependency structure forces the exact same accumulation order no matter how rows are physically distributed across ranks — the algorithm itself removes the degree of freedom that causes drift elsewhere.
