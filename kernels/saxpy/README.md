# SAXPY

**The algorithm.** SAXPY — "Single-precision A·X Plus Y," a classic BLAS Level-1 operation: `y[i] = a*x[i] + y[i]` for every element. The simplest possible elementwise vector combine — one scalar multiply, one add, per element, with no dependency between elements at all.

**The decomposition strategy — deliberately the simplest kernel in the repo.** One GPU thread per array element: each thread reads `x[i]` and `y[i]`, writes `y[i] = a*x[i] + y[i]`. Every access is already contiguous and coalesced by construction — no shared memory, no tiling, no neighbor dependency to manage, unlike every other CUDA kernel here. That's intentional: per the build plan, SAXPY's entire purpose is a warm-up — validating the CUDA measurement loop (event timing, achieved-bandwidth-as-percent-of-peak) on the simplest possible kernel before attempting anything harder, not demonstrating an optimization technique.

One real correctness detail worth noting: unlike a read-only kernel, SAXPY **mutates `y` in place**, so across repeated timed repetitions, `y` must be freshly re-copied to the GPU before *every single* repetition — otherwise repetition 2 would compute `a*x` against an already-updated `y` from repetition 1, silently corrupting every repetition after the first. `x`, by contrast, never changes, so it's copied once, outside the timed loop. (Also note: `-r` here means the *normal* thing — repeated timed runs — unlike jacobi's repurposed "iteration count" meaning next door.)

**What was measured.**

| Precision | N | Block | GFLOP/s | Bandwidth | % of peak |
|---|---|---|---|---|---|
| float | 50,000,000 | 256 | 14.32 | 85.92 GB/s | 103.95% |
| double | 25,000,000 | 256 | 7.17 | 86.02 GB/s | 104.06% |

**What was surprising.**
- Both precisions land right around **104% of peak** — the same small overshoot pattern and magnitude as reduction's `reduce_shuffle`, attributed to the same cause: `bandwidthTest`'s reference figure itself varies under GPU Boost, so a well-tuned kernel run can measure slightly above a peak captured in an earlier, separate run. This is a different (and much smaller) story than jacobi's >200% overshoot, which has no L2-cache-reuse explanation available here — SAXPY touches each element exactly once, with no neighbor overlap to be cached.
- The build plan's own Phase 4 D1 entry expected 70–85% of peak for this warm-up kernel, explicitly flagging that ~20% would mean the grid sizing was wrong. Landing at 103–104% — comfortably above even the optimistic end of that range — is a good early signal that the measurement loop and grid/block sizing were correct from the first CUDA kernel written, not a borderline pass.
