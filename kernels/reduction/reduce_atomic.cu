/*
 * Reduction, CUDA variant 1 of 3: global atomicAdd (deliberately naive).
 *
 * Phase 4, D2 -- "the highest-value kernel in the set" per the plan.
 * This file is the intentionally-slow baseline: every one of the N
 * threads adds its own element directly into ONE shared global-memory
 * location, using atomicAdd to make each individual add safe. The
 * next two files (reduce_shared.cu, reduce_shuffle.cu) exist purely
 * to fix the bottleneck this file deliberately doesn't avoid --
 * comparing all three timings later is the actual point of D2.
 *
 * -t means threads-per-block here, same repurposing as saxpy.cu.
 * Pass it explicitly (e.g. -t 256).
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
 * atomicAdd(result, value) is a single uninterruptible read-modify-
 * write: it reads *result, adds value, and writes the sum back, as
 * one hardware-guaranteed-safe operation. Without it, two threads
 * both reading *result at the same instant, both computing their own
 * "old + mine", and both writing back would silently lose one
 * thread's contribution (a classic race condition). The cost: every
 * thread wanting to touch *this same* memory location has to wait its
 * turn -- with N threads all targeting ONE location, they effectively
 * queue up single-file, which is exactly why this is the slow,
 * "naive" variant rather than the fast one.
 *
 * Double-precision atomicAdd requires compute capability 6.0+; the
 * P2000 is sm_61, so this compiles and runs with no workaround.
 */
__global__ void reduce_atomic_kernel(int n, const real *x, real *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, x[idx]);
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

    const int n = cfg.n;
    const int threads_per_block = cfg.threads;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    real *h_x = (real *)malloc(n * sizeof(real));
    if (!h_x) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* seed=42 matches reference/check_reduction.py exactly. Note this
     * oracle comparison is only meaningful at -n 1024 (the oracle's
     * fixed correctness-check size) -- larger N values are for
     * benchmarking only and won't have a matching expected sum,
     * exactly like matmul's separate correctness-check vs.
     * benchmark-cap sizes. */
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
        /* Reset the accumulator to zero before every launch, timed or
         * not -- same in-place-mutation problem as saxpy.cu's y, just
         * a single accumulator instead of a whole array. cudaMemset
         * writes raw zero BYTES, which is safe here specifically
         * because IEEE-754 positive zero is represented as all-zero
         * bits for both float and double -- this trick would NOT be
         * safe for, say, setting a float to 1.0. Kept outside the
         * timed region, same reasoning as saxpy.cu's y-reset copy. */
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(real)));

        CUDA_CHECK(cudaEventRecord(start));
        reduce_atomic_kernel<<<blocks, threads_per_block>>>(n, d_x, d_result);
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

    /* n additions total; memory traffic is dominated by reading x
     * (n elements) -- the atomic writes all target one single shared
     * location, so they don't scale with n the way a full array
     * read/write would. */
    double gflops   = (double)n / time_min / 1e9;
    double gbytes_s = ((double)n * sizeof(real)) / time_min / 1e9;

    bench_report_csv(stdout, "reduction", "cuda_atomic", n, threads_per_block,
                      time_min, time_med, gflops, gbytes_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_result));
    free(h_x);
    free(rep_times);

    return 0;
}
