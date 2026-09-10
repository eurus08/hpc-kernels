/*
 * Transpose, CUDA variant 1 of 2: naive (no shared memory).
 *
 * Phase 4, D3. First CUDA file in the repo to use a 2D thread grid --
 * every prior kernel (saxpy, reduction) worked on flat 1D arrays with
 * a single index; a matrix naturally maps to (row, col), so threads
 * are launched in a 2D block/grid using threadIdx.x/y and
 * blockIdx.x/y directly, instead of computing one flat index.
 *
 * -t is repurposed AGAIN here: it means the tile dimension (a square
 * TILE x TILE block of threads), not a flat thread count or threads-
 * per-block as in saxpy.cu/reduce_*.cu. -t 32 gives a 32x32 = 1024
 * thread block (the hardware's per-block maximum, and exactly one
 * warp's width in each dimension).
 *
 * This naive version reproduces the SAME asymmetric-locality problem
 * already documented for kernels/transpose/serial.c (good read
 * pattern, bad write pattern -- or vice versa, depending on loop/
 * index order) -- except on the GPU this isn't just a cache-locality
 * issue, it's a COALESCING issue: neighboring threads reading or
 * writing neighboring addresses lets the hardware fetch/store it all
 * in one efficient transaction; a strided access pattern instead
 * costs one separate transaction per thread. transpose_tiled.cu
 * (next file) exists specifically to fix the uncoalesced direction
 * this file leaves broken.
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
 * One thread per element. row/col come directly from the 2D block/
 * grid position -- no flat-index arithmetic needed, unlike every
 * prior 1D kernel.
 *
 * Reading A[row*n + col]: as threadIdx.x increases (col increases),
 * consecutive threads read CONSECUTIVE addresses -- coalesced, one
 * efficient transaction.
 *
 * Writing B[col*n + row]: as threadIdx.x increases (col increases),
 * consecutive threads write addresses n elements APART (each step in
 * col jumps a full row's width in B) -- NOT coalesced, one separate
 * transaction per thread. This is the asymmetry this file
 * deliberately leaves unfixed.
 */
__global__ void transpose_naive_kernel(int n, const real *A, real *B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        B[col * n + row] = A[row * n + col];
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
    const int tile = cfg.threads; /* repurposed: square tile/block dim */

    if (tile < 1 || tile > 32) {
        fprintf(stderr,
                "error: -t %d (tile dimension) must be between 1 and 32 "
                "-- tile*tile threads/block cannot exceed the hardware "
                "limit of 1024\n", tile);
        return 1;
    }

    /* 2D grid: enough TILE x TILE blocks in each dimension to cover
     * the full n x n matrix, rounding up same as every prior kernel's
     * 1D grid sizing. */
    dim3 block_dim(tile, tile);
    dim3 grid_dim((n + tile - 1) / tile, (n + tile - 1) / tile);

    real *h_A = (real *)malloc((size_t)n * n * sizeof(real));
    real *h_B = (real *)malloc((size_t)n * n * sizeof(real));
    if (!h_A || !h_B) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* seed=42, matching reference/check_transpose.py and the existing
     * CPU serial.c/omp.c exactly -- same oracle-comparison caveat as
     * every prior kernel: only meaningful at -n 512, the oracle's
     * fixed correctness-check size. */
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

    /* Unlike saxpy.cu/reduce_*.cu, B is fully OVERWRITTEN every
     * launch (every element gets a fresh write, nothing accumulates
     * or depends on B's previous contents) -- so there is no
     * in-place-mutation problem here, and no reset-before-each-rep
     * step is needed. */
    for (int r = 0; r < warmups + cfg.reps; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        transpose_naive_kernel<<<grid_dim, block_dim>>>(n, d_A, d_B);
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

    /* Same sanity-check convention as kernels/transpose/serial.c:
     * NOT a sum-based check (transpose only rearranges values, never
     * changes which values exist, so sum(B) == sum(A) even for a
     * badly broken kernel) -- print one specific off-diagonal pair
     * instead, which must match exactly if correct. */
    printf("sanity: B[0][1] = " REAL_FMT ", A[1][0] = " REAL_FMT "\n",
           h_B[0 * n + 1], h_A[1 * n + 0]);

    /* Pure data movement: 0 FLOPs. One read of A, one write of B,
     * both full n*n matrices. */
    double gflops   = 0.0;
    double gbytes_s = (2.0 * n * n * sizeof(real)) / time_min / 1e9;

    bench_report_csv(stdout, "transpose", "cuda_naive", n, tile,
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
