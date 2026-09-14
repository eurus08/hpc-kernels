/*
 * Reduction, CUDA variant 2 of 3: shared-memory tree reduction.
 *
 * Phase 4, D2. Fixes reduce_atomic.cu's bottleneck (N threads all
 * fighting over one global accumulator) by having each BLOCK first
 * combine its own threads' values locally, in fast on-chip shared
 * memory, down to one partial sum -- only THEN does exactly one
 * atomicAdd per block (not per thread) touch the global accumulator.
 * With 256 threads/block, that's N/256 global atomics instead of N.
 *
 * Three techniques combined here, each explained where it appears:
 *   1. Sequential-addressing tree (contiguous active-thread ranges at
 *      every step -- avoids shared-memory bank conflicts).
 *   2. "First add during the global load" -- each thread loads AND
 *      pre-adds two elements before the tree even starts, halving
 *      the number of blocks needed for a given N.
 *   3. Exactly one atomicAdd per block at the end, keeping this
 *      directly timing-comparable to reduce_atomic.cu (one kernel,
 *      one launch, no extra host-side combine step).
 *
 * threads_per_block MUST be a power of 2 -- the tree-halving logic
 * (s >>= 1 each step) requires it; enforced with a hard check below.
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

static int is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

/*
 * `extern __shared__ real sdata[]` declares a shared-memory array
 * whose SIZE is decided at launch time, not compile time (the third
 * argument in the <<<blocks, threads, SIZE>>> launch below) -- needed
 * here because threads_per_block comes from the -t CLI flag, not a
 * fixed constant.
 */
extern __shared__ real sdata[];

__global__ void reduce_shared_kernel(int n, const real *x, real *result) {
    int tid = threadIdx.x;

    /* Each thread is responsible for TWO elements, spaced blockDim.x
     * apart, added together right here during the load -- this is
     * the "first add during the global load" step. Both reads need
     * their own bounds check since n may not be an exact multiple of
     * (2 * blockDim.x * gridDim.x). */
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    real my_sum = 0;
    if (i < n) {
        my_sum = x[i];
    }
    if (i + blockDim.x < n) {
        my_sum += x[i + blockDim.x];
    }
    sdata[tid] = my_sum;

    /* __syncthreads() is a barrier: every thread in this block stops
     * here until ALL of them have arrived. Required because the next
     * tree step reads sdata[] positions that OTHER threads just
     * wrote -- without this barrier, a fast thread could read a
     * neighbor's shared-memory slot before that neighbor finished
     * writing it, an undefined-behavior race condition. */
    __syncthreads();

    /* Sequential-addressing tree: s halves each step (256 -> 128 ->
     * 64 -> ... -> 1), and at every step the ACTIVE thread IDs
     * (tid < s) stay a single contiguous block (0..s-1), not a
     * strided pattern -- this is what avoids shared-memory bank
     * conflicts (32 threads hitting the same one of 32 memory
     * "banks" simultaneously, which would serialize instead of
     * running in parallel). */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Exactly one atomicAdd per BLOCK (not per thread) -- only
     * thread 0 executes this, with sdata[0] now holding this whole
     * block's combined sum. */
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
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
        fprintf(stderr, "usage: %s -n <size> [-r <reps>] [-t <threads_per_block>]\n",
                argv[0]);
        return 1;
    }

    if (!is_power_of_two(cfg.threads)) {
        fprintf(stderr,
                "error: -t %d must be a power of 2 for the tree reduction "
                "(e.g. 128, 256, 512)\n", cfg.threads);
        return 1;
    }

    const int n = cfg.n;
    const int threads_per_block = cfg.threads;
    /* Each block now handles 2 * threads_per_block elements (the
     * "first add during load" halving), so half as many blocks are
     * needed compared to reduce_atomic.cu's one-element-per-thread
     * grid sizing. */
    const int elems_per_block = 2 * threads_per_block;
    const int blocks = (n + elems_per_block - 1) / elems_per_block;
    const size_t shared_mem_bytes = threads_per_block * sizeof(real);

    real *h_x = (real *)malloc(n * sizeof(real));
    if (!h_x) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* seed=42, matching reference/check_reduction.py -- same oracle-
     * comparison caveat as reduce_atomic.cu: only meaningful at
     * -n 1024, the oracle's fixed correctness-check size. */
    genmat_random(h_x, n, 1, 42u);

    real *d_x, *d_result;
    CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(real)));
    CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(real)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(real), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int warmups = 2;
    double *rep_times = (double *)malloc(cfg.reps * sizeof(double));
    if (!rep_times) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int r = 0; r < warmups + cfg.reps; r++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(real)));

        CUDA_CHECK(cudaEventRecord(start));
        reduce_shared_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
            n, d_x, d_result);
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

    real h_result = 0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(real), cudaMemcpyDeviceToHost));

    fprintf(stderr, "sanity: sum = " REAL_FMT "\n", h_result);

    double gflops   = (double)n / time_min / 1e9;
    double gbytes_s = ((double)n * sizeof(real)) / time_min / 1e9;

    bench_report_csv(stdout, "reduction", "cuda_shared", n, threads_per_block,
                      time_min, time_med, gflops, gbytes_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_result));
    free(h_x);
    free(rep_times);

    return 0;
}
