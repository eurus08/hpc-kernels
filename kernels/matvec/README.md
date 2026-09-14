# Matvec

**The algorithm.** Dense matrix-vector multiply, `y = A × x` — an N×N matrix times a length-N vector, producing a length-N result. Each output element `y[i]` is one row of `A` dotted with `x`; O(N²) work total, one order of magnitude less than matmul's O(N³) for the same N.

**The decomposition strategy.**
- **`serial.c`** — one loop over rows, each computing its own dot product with `x`. Baseline.
- **`mpi.c` — row-distributed, with a communication-avoidance trick worth calling out.** `A`'s rows are split into contiguous blocks across ranks, balanced as evenly as possible (no rank differs from another by more than one row), and it degrades gracefully if there are more ranks than rows — extra ranks simply get zero rows and sit idle, no special-case code needed. **`x` is never communicated at all**: since `genmat_random()` is a deterministic PRNG, every rank independently regenerates the *entire* `x` vector from the same seed and ends up with a bit-identical copy — zero messages, instead of rank 0 generating it and broadcasting. (This trick only works for `x`, not for splitting `A`'s generation the same way — `A`'s PRNG stream is sequential, so asking rank 2 to "generate its own slice" would replay the start of the stream, not the correct offset. `A` genuinely has to be generated whole on rank 0, then physically scattered.) Because rows are contiguous in row-major storage, distributing `A` needs only a plain `MPI_Scatterv` — no derived 2D datatype like Cannon's algorithm requires. After that one-time scatter, no further communication happens during compute at all — each rank computes its own rows' worth of `y` completely independently. Verification only needs `sum(y)`, collected via a single final `MPI_Reduce` — never a full gather of the result vector.

**What was measured.**

*MPI rank scaling, N=6144 (speedup / efficiency):*
| Ranks | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Speedup | 1.00× | 1.71× | 3.20× | 4.32× | 5.72× |
| Efficiency | 100% | 85.71% | 79.91% | 71.93% | 71.44% |

**What was surprising.**
- Efficiency holds up far better here than in either other MPI kernel in this repo — 71.44% at 8 ranks, versus reduction's 20.26% at 8 ranks and matmul's 15.92% at 9. The difference is structural, not incidental: matvec communicates *once* (the initial scatter) and then runs fully independently, while Cannon's algorithm re-communicates every round and reduction re-combines on every call.
- No sharp hyperthreading cliff at 8 ranks (71.44%, barely below 6 ranks' 71.93%), unlike matmul's steep drop at the same worker counts. Makes sense given matvec has no tight communication rounds for oversubscribed ranks to collide over — the workload is close to embarrassingly parallel once the one-time scatter is done.
