/*
 * SAXPY: y = a*x + y
 *
 * Phase 4, D1 -- the CUDA warm-up kernel. The point of this file isn't
 * the math (it's one line) -- it's establishing the measurement loop
 * for everything that follows in Phase 4: cudaEvent-based timing,
 * host<->device memory transfers, and comparing achieved bandwidth
 * against the Phase 0 bandwidthTest figure (72,373.6 MB/s measured).
 *
 * -t is repurposed here to mean "threads per block" (a CUDA-specific
 * concept), not CPU thread count / MPI rank count as in every prior
 * file -- still fits the CSV's threads_or_ranks column conceptually.
 * ALWAYS pass -t explicitly (e.g. -t 256) -- bench_parse_args defaults
 * -t to 1 when omitted, which would launch one thread per block: a
 * legal but extremely poor choice on a GPU.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "genmat.h"
#include "types.h"

/*
 * Every CUDA runtime call returns a cudaError_t instead of throwing an
 * exception. If you don't check it, a failure (e.g. asking for more
 * VRAM than exists) can go unnoticed and either crash somewhere much
 * later with a confusing unrelated error, or silently produce wrong
 * results. This macro wraps a call, checks its return code, and if
 * anything went wrong, prints a human-readable message (cudaGetErrorString)
 * and exits immediately -- fail loud, fail at the source.
 */
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
 * The kernel itself: what ONE GPU thread does. Each thread computes
 * exactly one element of y. `idx` identifies which element this
 * particular thread is responsible for, derived from CUDA's built-in
 * per-block/per-thread IDs. The bounds check matters because the
 * number of threads launched (blocks * threadsPerBlock) is rounded UP
 * to cover n -- the last block may have a few threads with no real
 * element to work on, and without the check they'd read/write past
 * the end of the array.
 */
__global__ void saxpy_kernel(int n, real a, const real *x, real *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

/* Ascending-order comparator for qsort, used to find the median timed
 * repetition below. */
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
    const real a = (real)2.5;  /* matches reference/check_saxpy.py exactly */
    const int threads_per_block = cfg.threads;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    /* Host-side buffers. y_orig holds the untouched original y so it
     * can be re-copied to the GPU before every repetition -- SAXPY
     * overwrites y using its own old value, so without resetting it,
     * repetition 2 would compute a*x + (already-updated y) instead of
     * the same thing repetition 1 computed, corrupting both the
     * correctness check and the timing comparison across reps. */
    real *h_x     = (real *)malloc(n * sizeof(real));
    real *h_y_orig = (real *)malloc(n * sizeof(real));
    real *h_y     = (real *)malloc(n * sizeof(real));
    if (!h_x || !h_y_orig || !h_y) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    /* Seeds 60/61 match reference/check_saxpy.py exactly -- same
     * PRNG, same seeds, same sequence, so the oracle's expected[]
     * is checking against the identical input this kernel sees. */
    genmat_random(h_x, n, 1, 60u);
    genmat_random(h_y_orig, n, 1, 61u);

    /* Device-side buffers. */
    real *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(real)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, n * sizeof(real)));

    /* x never changes across repetitions -- copy it once, outside any
     * timed region. */
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(real), cudaMemcpyHostToDevice));

    /* cudaEvent_t is CUDA's timing mechanism -- the GPU equivalent of
     * MPI_Wtime()/clock_gettime() used everywhere so far. You place an
     * event marker immediately before and after the work you want to
     * time, then ask CUDA for the elapsed milliseconds between them.
     * This measures GPU execution time directly, not CPU wall-clock
     * time spent waiting on the GPU. */
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
        /* Reset y on the device to its original value before every
         * launch, including warm-ups. This copy is deliberately
         * OUTSIDE the timed region -- we're timing the kernel's
         * compute+memory-access performance, comparable against the
         * Phase 0 bandwidthTest figure, not this reset transfer. */
        CUDA_CHECK(cudaMemcpy(d_y, h_y_orig, n * sizeof(real),
                               cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        saxpy_kernel<<<blocks, threads_per_block>>>(n, a, d_x, d_y);
        CUDA_CHECK(cudaEventRecord(stop));

        /* cudaEventSynchronize blocks the CPU until the GPU has
         * actually reached the `stop` marker -- kernel launches are
         * asynchronous (the CPU would otherwise race ahead and read a
         * meaningless "elapsed time" before the GPU finished). */
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        if (r >= warmups) {
            rep_times[r - warmups] = (double)ms / 1000.0;  /* ms -> seconds */
        }
    }

    /* Check for any launch-time error (e.g. illegal grid config) that
     * the per-call CUDA_CHECK macros above wouldn't necessarily catch,
     * since kernel launches don't return an error code directly --
     * errors from the launch itself surface on the next CUDA call. */
    CUDA_CHECK(cudaGetLastError());

    /* min and median across the timed repetitions -- same discipline
     * as every prior kernel's bench_run, reimplemented by hand here
     * because bench_run assumes a single-core CPU kernel_fn, not a
     * GPU kernel launch. */
    qsort(rep_times, cfg.reps, sizeof(double), compare_double);
    double time_min = rep_times[0];
    double time_med = rep_times[cfg.reps / 2];

    /* Copy the final result back for the correctness check. */
    CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(real), cudaMemcpyDeviceToHost));

    fprintf(stderr, "sanity: y[0] = " REAL_FMT "\n", h_y[0]);

    /* SAXPY moves 3*n elements of memory (read x, read y, write y)
     * and performs 2*n floating-point ops (one multiply, one add per
     * element). */
    double gflops  = (2.0 * n) / time_min / 1e9;
    double gbytes_s = (3.0 * n * sizeof(real)) / time_min / 1e9;

    bench_report_csv(stdout, "saxpy", "cuda", n, threads_per_block,
                      time_min, time_med, gflops, gbytes_s);

    /* Cleanup. */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y_orig);
    free(h_y);
    free(rep_times);

    return 0;
}
