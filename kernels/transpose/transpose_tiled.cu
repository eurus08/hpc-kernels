/*
 * Transpose, CUDA variant 2 of 2: shared-memory tiled.
 *
 * Phase 4, D3, the fix for transpose_naive.cu's uncoalesced write.
 * The trick: instead of ever letting a thread write directly to a
 * strided global-memory address, each block first stages a TILE x
 * TILE chunk of A into fast on-chip shared memory (one coalesced
 * read), then writes that chunk back out to B's transposed location
 * -- ALSO coalesced -- by reading the tile with swapped (x,y) indices
 * INSIDE shared memory (cheap) instead of swapping which global
 * address each thread touches (expensive/uncoalesced).
 *
 * Two block-grid-index swaps happen together, easy to conflate, so
 * spelled out precisely at each site below:
 *   1. Which TILE of the grid a block is responsible for writing to
 *      swaps (blockIdx.x and blockIdx.y trade roles between the load
 *      and store phases).
 *   2. Which THREAD's value ends up at a given shared-memory slot
 *      also swaps (tile[tx][ty] instead of tile[ty][tx]) during the
 *      store phase -- this is the actual transpose; step 1 alone
 *      would just copy tiles to new block positions without
 *      transposing their contents.
 *
 * Shared memory is declared with a "+1" padding column -- explained
 * in detail at the padding site below, since it's a genuinely
 * different reasoning trigger than D2's sequential-addressing fix,
 * despite both being called "bank conflict avoidance."
 *
 * -t means tile dimension, same repurposing as transpose_naive.cu.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "genmat.h"
#include "types.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/*
 * Declared as a flat 1D array, not a 2D [TILE][TILE+1] array, because
 * TILE is a runtime value (from -t), not a compile-time constant --
 * a static 2D shared-memory array needs its dimensions fixed at
 * compile time, but `extern __shared__` with manual row-stride
 * arithmetic works for any block size decided at launch.
 *
 * tile_w (= blockDim.x + 1, computed in the kernel) is that manual
 * row stride -- the "+1" padding column. A row of the tile is
 * tile_w elements wide in memory, even though only blockDim.x of
 * them hold real data; the extra element per row is never read, it
 * exists purely to shift memory addresses.
 */
extern __shared__ real tile[];

__global__ void transpose_tiled_kernel(int n, const real *A, real *B) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_w = blockDim.x + 1; /* the padded row stride */

    /* --- Load phase: stage one TILE x TILE chunk of A into shared
     * memory. col increases with tx, so consecutive threads read
     * consecutive A addresses -- coalesced, same as the "good"
     * direction transpose_naive.cu already had. */
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    if (row < n && col < n) {
        tile[ty * tile_w + tx] = A[row * n + col];
    }

    /* Barrier: every thread in the block must finish loading before
     * any thread starts reading a DIFFERENT thread's slot in the
     * next phase (same reasoning as D2's tree reduction). */
    __syncthreads();

    /* --- Store phase: block-index swap (point 1 above). blockIdx.y
     * (this block's ROW position when loading) now determines the
     * output COLUMN block, and blockIdx.x (the load's column
     * position) now determines the output ROW block -- this is what
     * sends each tile to its transposed location in the grid. */
    int out_col = blockIdx.y * blockDim.y + tx;
    int out_row = blockIdx.x * blockDim.x + ty;

    if (out_row < n && out_col < n) {
        /*
         * Thread-index swap (point 2 above): reading tile[tx][ty]
         * instead of tile[ty][tx] is what actually transposes the
         * DATA, not just its block location -- without this swap,
         * step 1 alone would just copy each tile verbatim to a new
         * position, which is not a transpose.
         *
         * Why the "+1" padding matters, concretely: consider the 32
         * threads of one warp during this read, all sharing the
         * same ty (fixed) with tx = 0..31 (consecutive). Each reads
         * address tx*tile_w + ty. Without padding (tile_w == 32,
         * matching a 32-wide tile exactly), that address MODULO 32
         * (there are 32 shared-memory banks, each handling one
         * distinct address-mod-32 value per cycle) is
         * (tx*32 + ty) mod 32 == ty mod 32 -- IDENTICAL for every
         * value of tx. All 32 threads hit the exact same bank
         * simultaneously: a 32-way bank conflict, serialized down to
         * one thread at a time. With tile_w = 33 (the "+1"), the
         * address mod 32 becomes (tx*33 + ty) mod 32 == (tx + ty)
         * mod 32 -- a DIFFERENT value for each tx in the warp, so
         * all 32 threads land on distinct banks and run in parallel,
         * exactly like D2's sequential addressing avoided conflicts
         * by keeping active thread IDs contiguous rather than
         * strided.
         */
        B[out_row * n + out_col] = tile[tx * tile_w + ty];
    }
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int main(int argc, char **argv) {
    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        fprintf(stderr, "usage: %s -n <size> [-r <reps>] [-t <tile_dim>]\n",
                argv[0]);
        return 1;
    }

    const int n = cfg.n;
    const int tile_dim = cfg.threads;

    if (tile_dim < 1 || tile_dim > 32) {
        fprintf(stderr,
                "error: -t %d (tile dimension) must be between 1 and 32 "
                "-- tile*tile threads/block cannot exceed the hardware "
                "limit of 1024\n", tile_dim);
        return 1;
    }

    dim3 block_dim(tile_dim, tile_dim);
    dim3 grid_dim((n + tile_dim - 1) / tile_dim, (n + tile_dim - 1) / tile_dim);
    const size_t shared_mem_bytes = (size_t)tile_dim * (tile_dim + 1) * sizeof(real);

    real *h_A = (real *)malloc((size_t)n * n * sizeof(real));
    real *h_B = (real *)malloc((size_t)n * n * sizeof(real));
    if (!h_A || !h_B) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* seed=42, matching reference/check_transpose.py, identical to
     * transpose_naive.cu -- same input, so both variants must agree
     * on B[0][1]/A[1][0] exactly. */
    genmat_random(h_A, n, n, 42u);

    real *d_A, *d_B;
    CUDA_CHECK(cudaMalloc((void **)&d_A, (size_t)n * n * sizeof(real)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, (size_t)n * n * sizeof(real)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)n * n * sizeof(real),
                           cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int warmups = 2;
    double *rep_times = (double *)malloc(cfg.reps * sizeof(double));
    if (!rep_times) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Same as transpose_naive.cu: B is fully overwritten every
     * launch, no reset-before-rep step needed. */
    for (int r = 0; r < warmups + cfg.reps; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        transpose_tiled_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
            n, d_A, d_B);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        if (r >= warmups) {
            rep_times[r - warmups] = (double)ms / 1000.0;
        }
    }

    CUDA_CHECK(cudaGetLastError());

    qsort(rep_times, cfg.reps, sizeof(double), compare_double);
    double time_min = rep_times[0];
    double time_med = rep_times[cfg.reps / 2];

    CUDA_CHECK(cudaMemcpy(h_B, d_B, (size_t)n * n * sizeof(real),
                           cudaMemcpyDeviceToHost));

    fprintf(stderr, "sanity: B[0][1] = " REAL_FMT ", A[1][0] = " REAL_FMT "\n",
           h_B[0 * n + 1], h_A[1 * n + 0]);

    double gflops   = 0.0;
    double gbytes_s = (2.0 * n * n * sizeof(real)) / time_min / 1e9;

    bench_report_csv(stdout, "transpose", "cuda_tiled", n, tile_dim,
                      time_min, time_med, gflops, gbytes_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);
    free(rep_times);

    return 0;
}
