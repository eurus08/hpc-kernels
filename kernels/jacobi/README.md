# Jacobi

**The algorithm.** 2D Jacobi iteration, 5-point stencil: each interior grid point is updated as a function of its previous value plus its four direct neighbors (up/down/left/right) — the standard iterative method for solving discretized PDEs like the Laplace/Poisson equation. CUDA-only in this repo (per the completeness table), included as Phase 4's D4 item, run last since it depended on D1–D3 finishing cleanly first.

**The decomposition strategy — no shared memory, deliberately.** Unlike `transpose_tiled.cu`, this kernel uses no shared-memory staging at all: since each thread's stencil only touches its four immediate neighbors, straightforward flat indexing (thread `(x,y)` maps directly to grid point `(x,y)`) already gives coalesced global-memory access on its own — there's no strided-access problem here to engineer around the way transpose has.

The one new mechanism this kernel introduces: **double buffering via pointer swap.** Every new grid value must be computed *entirely* from the *previous* iteration's grid — but with thousands of GPU threads running at slightly different speeds, there's no way to guarantee "read old, write new" ordering within a single array (a fast thread could read a neighbor's value after that neighbor already overwrote it for the new iteration). The fix: keep two full grids, always read from one and write to the other, then swap which pointer is "old" and which is "new" after each iteration — a cheap host-side pointer swap, not a data copy.

**A real CLI-semantics quirk, worth documenting explicitly rather than silently:** `-r` means something different here than in every other kernel in this repo. Everywhere else, `-r` is the number of *repeated timed runs* of a benchmark (with warm-up iterations discarded). Here, `-r` is repurposed to mean the **fixed iteration count of the Jacobi sweep itself** — the whole multi-iteration solve is timed as one unit, once, since a single Jacobi run already *is* many iterations, and that matches how the reference oracle runs one fixed-length sweep rather than several repeated ones.

**What was measured.**

| Precision | N (interior) | Block | GFLOP/s | Bandwidth | % of peak |
|---|---|---|---|---|---|
| float | 4096 | 16×16 | 43.09 | 172.36 GB/s | 208.5% |
| double | 2048 | 16×16 | 25.37 | 202.96 GB/s | 245.5% |

**What was surprising.** Both figures are far over 100% of the measured peak bandwidth — much larger than reduction's ~104% overshoot, and a different underlying cause. Investigated rather than dismissed: the 5-point stencil's access pattern means neighboring points' neighbor-sets overlap heavily, so a large share of each point's "reads" are likely served from L2 cache rather than round-tripping to DRAM. `bench.c`'s bandwidth formula counts *algorithmic* bytes touched (5 reads + 1 write per point, per iteration) rather than measuring actual DRAM traffic directly — so when cache reuse is this high, the reported figure exceeds the true DRAM-bound ceiling, since most of those "touches" never leave the chip at all. **This is inference from the data pattern, not confirmed via direct hardware profiling** (e.g. `nsight`/`nvprof` memory counters) — stated here as the likely explanation, not a settled one.
