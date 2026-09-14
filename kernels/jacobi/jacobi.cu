/*
 * Jacobi 2D 5-point stencil.
 *
 * Phase 4, D4 -- CUDA-only, included since D1-D3 all finished cleanly.
 * "One kernel, no optimization beyond coalesced access" per the plan:
 * unlike transpose_tiled.cu, this file uses NO shared memory -- the
 * only performance concern here is making sure neighboring threads
 * touch neighboring memory addresses, which the indexing below
 * achieves directly, with nothing extra to engineer around it.
 *
 * New mechanism: double buffering via pointer swap. Every new grid
 * value must be computed ENTIRELY from the previous iteration's grid
 * -- with thousands of threads running at different speeds, there's
 * no way to guarantee "read old, write new" ordering within a single
 * array (a fast thread could read a slower neighbor's value after
 * that neighbor already overwrote it). Fix: keep two full grids,
 * always read from one and write to the other, then swap which
 * pointer is "old" and which is "new" after each iteration -- a
 * cheap host-side pointer swap, not a data copy.
 *
 * -t means block dimension per axis (a tile-shaped 2D block, same
 * repurposing as transpose_*.cu), e.g. -t 16 gives a 16x16 block.
 * -r (repetitions, from bench_config) is repurposed to mean the
 * FIXED ITERATION COUNT of the Jacobi sweep itself, not repeated
 * timed runs of a whole kernel launch -- there is no separate
 * "warm-up vs. timed" concept here, since one Jacobi run already
 * means many iterations of the same kernel; the whole ITERS-iteration
 * loop is timed as one unit, once, matching how the oracle itself
 * runs a single fixed-length sweep.
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
 * n = interior dimension; the full array is (n+2) x (n+2), M = n+2,
 * to hold a 1-cell-wide halo of fixed boundary on every side.
 * row/col are computed with a "+1" offset so thread (0,0) lands on
 * interior position (1,1), never touching the boundary at index 0.
 */
__global__ void jacobi_kernel(int n, int m, const real *old_grid, real *new_grid) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (row <= n && col <= n) {
        new_grid[row * m + col] = 0.25 * (
            old_grid[(row - 1) * m + col] +  /* up */
            old_grid[(row + 1) * m + col] +  /* down */
            old_grid[row * m + (col - 1)] +  /* left */
            old_grid[row * m + (col + 1)]    /* right */
        );
    }
    /* Threads mapping to boundary positions (row/col outside 1..n)
     * simply do nothing -- the boundary is never written by this
     * kernel, in either buffer, for the entire run. */
}

int main(int argc, char **argv) {
    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        fprintf(stderr, "usage: %s -n <interior_size> -r <iterations> [-t <block_dim>]\n",
                argv[0]);
        return 1;
    }

    const int n = cfg.n;               /* interior dimension */
    const int iters = cfg.reps;        /* repurposed: fixed sweep count */
    const int block_dim_size = cfg.threads;
    const int m = n + 2;               /* full dimension, with halo */

    if (block_dim_size < 1 || block_dim_size > 32) {
        fprintf(stderr,
                "error: -t %d (block dimension) must be between 1 and 32\n",
                block_dim_size);
        return 1;
    }

    dim3 block_dim(block_dim_size, block_dim_size);
    dim3 grid_dim((n + block_dim_size - 1) / block_dim_size,
                  (n + block_dim_size - 1) / block_dim_size);

    const size_t full_bytes = (size_t)m * m * sizeof(real);

    /* Host-side full grid: zero everywhere (including the boundary,
     * which stays zero permanently), random interior. */
    real *h_grid = (real *)calloc((size_t)m * m, sizeof(real));
    if (!h_grid) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* seed=70, matching reference/check_jacobi.py exactly. Fills only
     * the interior (n x n) region -- genmat_random doesn't know about
     * the halo layout, so we generate into a temporary flat buffer,
     * then place each row into its correct offset within h_grid
     * (skipping the boundary column on each side). */
    real *h_interior = (real *)malloc((size_t)n * n * sizeof(real));
    if (!h_interior) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    genmat_random(h_interior, n, n, 70u);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_grid[(i + 1) * m + (j + 1)] = h_interior[i * n + j];
        }
    }
    free(h_interior);

    real *d_a, *d_b;
    CUDA_CHECK(cudaMalloc((void **)&d_a, full_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, full_bytes));

    /* d_a: full initial state (boundary=0, interior=random), copied
     * from the host. d_b: zeroed entirely -- its boundary (0) is
     * correct and permanent; its interior content doesn't matter
     * yet, since the first iteration will write fresh values into
     * every interior position of d_b before anything ever reads
     * them back. */
    CUDA_CHECK(cudaMemcpy(d_a, h_grid, full_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_b, 0, full_bytes));

    real *d_old = d_a;
    real *d_new = d_b;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* The entire ITERS-iteration sweep is timed as ONE unit -- there
     * is no separate warm-up/timed-repetition split here, unlike
     * every prior Phase 4 kernel. A single Jacobi "run" IS the fixed
     * number of iterations; timing anything less wouldn't match what
     * the oracle itself computed. */
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        jacobi_kernel<<<grid_dim, block_dim>>>(n, m, d_old, d_new);

        /* Pointer swap: cheap, host-side, no data movement. After
         * this line, d_old points at the buffer jacobi_kernel just
         * finished WRITING (this iteration's fresh results), and
         * d_new points at the buffer that is now safe to overwrite
         * on the next iteration. */
        real *tmp = d_old;
        d_old = d_new;
        d_new = tmp;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double time_total = (double)ms / 1000.0;

    /* After the loop, d_old holds the most recently written buffer
     * (the final swap always leaves the newest data in d_old) -- the
     * actual result of all `iters` sweeps. */
    CUDA_CHECK(cudaMemcpy(h_grid, d_old, full_bytes, cudaMemcpyDeviceToHost));

    /* Same fixed-point sanity-check convention as every prior kernel:
     * print one specific value, compare by eye against the oracle.
     * (256, 256) in 1-indexed interior coordinates, matching
     * check_jacobi.py exactly -- only meaningful at -n 512 -r 100,
     * the oracle's fixed correctness-check configuration. */
    int check_i = 256, check_j = 256;
    fprintf(stderr, "sanity: grid[%d][%d] = " REAL_FMT "\n", check_i, check_j,
           h_grid[check_i * m + check_j]);

    /* Per-iteration cost: n*n interior points, each doing 4 adds + 1
     * multiply (5 flops), reading 4 neighbors + writing 1 value (5
     * elements of memory traffic) -- multiplied by the number of
     * iterations for a total-run figure, then divided by the total
     * time for a rate. */
    double gflops = (5.0 * n * n * iters) / time_total / 1e9;
    double gbytes_s = (5.0 * n * n * iters * sizeof(real)) / time_total / 1e9;

    bench_report_csv(stdout, "jacobi", "cuda", n, block_dim_size,
                      time_total, time_total, gflops, gbytes_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_grid);

    return 0;
}
