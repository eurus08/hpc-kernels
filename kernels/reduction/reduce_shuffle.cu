/*
 * Reduction, CUDA variant 3 of 3: warp-shuffle reduction.
 *
 * Phase 4, D2, final variant. Identical to reduce_shared.cu through
 * the shared-memory tree, EXCEPT the tree now stops early, once only
 * one warp's worth of values (32) remains, and finishes the last few
 * steps using __shfl_down_sync instead of shared memory.
 *
 * Why this is safe/faster: the 32 threads of a single warp execute in
 * true lockstep on this hardware (SIMT -- literally the same
 * instruction, same clock cycle, across all 32 lanes), so they don't
 * need __syncthreads() to coordinate with each other the way the rest
 * of the block does. __shfl_down_sync(mask, val, offset) reads
 * another lane's register value DIRECTLY -- no shared memory, no
 * barrier -- which is the fastest possible way to move data between
 * threads on this hardware.
 *
 * threads_per_block MUST be a power of 2 AND >= 64 (need at least two
 * warps' worth so the manual s=32 fold below has a valid tid+32 to
 * read).
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

extern __shared__ real sdata[];

__global__ void reduce_shuffle_kernel(int n, const real *x, real *result) {
    int tid = threadIdx.x;

    /* Identical load-and-pre-add step to reduce_shared.cu. */
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    real my_sum = 0;
    if (i < n) {
        my_sum = x[i];
    }
    if (i + blockDim.x < n) {
        my_sum += x[i + blockDim.x];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    /* Same sequential-addressing tree as reduce_shared.cu, but the
     * loop condition is "s > 32" instead of "s > 0" -- it deliberately
     * STOPS once 64 values remain (two warps' worth), one level
     * earlier than reduce_shared.cu's tree. The final 64->32 fold
     * is done manually just below, combined with entering the warp
     * so no separate __syncthreads() is spent on it. */
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Only the first warp (threads 0-31) does anything from here. */
    if (tid < 32) {
        /* Manual fold of the last shared-memory step (64 values ->
         * 32): thread tid combines its own value with the one 32
         * lanes ahead, still via shared memory (this data was
         * written by the LAST __syncthreads()-guarded loop iteration
         * above, so it's already safely visible here). */
        real val = sdata[tid] + sdata[tid + 32];

        /* Now genuinely warp-only: 5 steps (32 -> 16 -> 8 -> 4 -> 2
         * -> 1), each thread grabbing a neighbor's register value
         * directly, no shared memory, no barrier. 0xffffffff is the
         * "mask" argument -- it tells the hardware all 32 lanes of
         * this warp are participating (required by newer CUDA
         * versions' safety model for warp-synchronous instructions,
         * even though every lane genuinely is active here). */
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            atomicAdd(result, val);
        }
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

    if (!is_power_of_two(cfg.threads) || cfg.threads < 64) {
        fprintf(stderr,
                "error: -t %d must be a power of 2 and >= 64 "
                "(e.g. 64, 128, 256, 512)\n", cfg.threads);
        return 1;
    }

    const int n = cfg.n;
    const int threads_per_block = cfg.threads;
    const int elems_per_block = 2 * threads_per_block;
    const int blocks = (n + elems_per_block - 1) / elems_per_block;
    const size_t shared_mem_bytes = threads_per_block * sizeof(real);

    real *h_x = (real *)malloc(n * sizeof(real));
    if (!h_x) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* seed=42, matching reference/check_reduction.py, same caveat as
     * the other two variants: only meaningful at -n 1024. */
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
        reduce_shuffle_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
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

    bench_report_csv(stdout, "reduction", "cuda_shuffle", n, threads_per_block,
                      time_min, time_med, gflops, gbytes_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_result));
    free(h_x);
    free(rep_times);

    return 0;
}
