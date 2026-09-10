/*
 * kernels/transpose/omp.c
 *
 * OpenMP matrix transpose: same out-of-place B = A^T as serial.c, with the
 * outer loop parallelized across threads via #pragma omp parallel for.
 *
 * Why this is safe with zero extra care: each thread's iterations of the
 * outer loop (each value of i) write to a completely disjoint set of B
 * locations -- for a fixed i, the inner loop touches B[0*n+i], B[1*n+i],
 * ..., B[(n-1)*n+i], and no other value of i ever writes to any of those
 * same addresses. No two threads can ever write the same element of B,
 * so there's no race condition and no need for any synchronization
 * (no critical section, no reduction, no atomic) -- this is about as
 * simple as OpenMP parallelization gets.
 *
 * Plain outer-loop parallelization only, not #pragma omp parallel for
 * collapse(2): collapse(2) would also be safe (same disjoint-write
 * argument applies to the flattened (i,j) iteration space too), but the
 * outer loop alone already gives every thread plenty of independent work
 * -- n row-strips split across a handful of cores -- so collapsing
 * wouldn't change anything meaningful here, and the plan specifically
 * describes parallelizing the outer loop. Confirmed with Prince before
 * writing this file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "types.h"
#include "genmat.h"
#include "bench.h"

typedef struct {
    const real *A;
    real *B;
    int n;
} transpose_arg;

static void transpose_kernel(void *arg)
{
    transpose_arg *a = (transpose_arg *)arg;
    const int n = a->n;
    const real *A = a->A;
    real *B = a->B;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[(size_t)j * n + i] = A[(size_t)i * n + j];
        }
    }
}

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s -n <size> [-r <reps>] [-t <threads>]\n", prog);
}

int main(int argc, char **argv)
{
    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        usage(argv[0]);
        return 1;
    }

    omp_set_num_threads(cfg.threads);

    const int n = cfg.n;

    real *A = malloc((size_t)n * n * sizeof(real));
    real *B = malloc((size_t)n * n * sizeof(real));
    if (!A || !B) {
        fprintf(stderr, "allocation failed for n=%d\n", n);
        free(A); free(B);
        return 1;
    }

    genmat_random(A, n, n, 42);

    transpose_arg arg = { A, B, n };

    double time_min, time_med;
    bench_run(transpose_kernel, &arg, &cfg, &time_min, &time_med);

    real b01 = (n >= 2) ? B[0 * n + 1] : (real)0.0;
    real a10 = (n >= 2) ? A[1 * n + 0] : (real)0.0;
    fprintf(stderr, "B[0][1]=%.7e A[1][0]=%.7e (should match exactly)\n",
            (double)b01, (double)a10);

    double gflops = 0.0;
    double gbytes_s = (2.0 * (double)n * (double)n) * sizeof(real) / time_min / 1e9;

    bench_report_csv(stdout, "transpose", "omp", n, cfg.threads,
                      time_min, time_med, gflops, gbytes_s);

    free(A);
    free(B);
    return 0;
}
