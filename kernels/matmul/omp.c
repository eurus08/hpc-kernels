/*
 * kernels/matmul/omp.c
 *
 * OpenMP (multi-core, single machine) dense matrix-matrix multiply:
 * C = A * B, for square N x N matrices. Same ikj loop ordering and
 * algorithm as kernels/matmul/serial.c -- the only change is that the
 * OUTER loop (i) is split across threads with a single #pragma.
 *
 * Beginner note on #pragma omp parallel for:
 * OpenMP's #pragma directives are hints to the compiler, ignored
 * entirely if you compile without -fopenmp (the code still works,
 * just runs on one core). `#pragma omp parallel for` tells the
 * compiler: split the iterations of the loop immediately below across
 * however many threads are available, each thread running a subset of
 * i values. With the default "static" schedule, that split is simply
 * contiguous chunks -- e.g. 4 threads on n=256 means thread 0 gets
 * i=0..63, thread 1 gets i=64..127, and so on. That's a good fit here
 * because every i does the same amount of work (a full row's worth of
 * k*j multiply-adds) -- no thread ends up starving while another is
 * still busy.
 *
 * Why this is safe with NO extra synchronization (no locks, no
 * critical sections, no explicit "private" clauses):
 *   - Each thread writes to a DIFFERENT set of rows of C (C[i*n+j] for
 *     its own i values only) -- two threads never touch the same
 *     memory, so there's no race to guard against.
 *   - k, j, and a_ik are all declared INSIDE the i-loop's body. In C,
 *     that means each execution of the loop body gets its own fresh
 *     copy of those variables on the stack -- OpenMP calls this
 *     "private by default for loop-local declarations," and it's
 *     exactly why no explicit private(k,j,a_ik) clause is needed here.
 *   - A and B are only ever READ, never written, so many threads
 *     reading the same memory simultaneously is completely fine.
 *
 * A subtler point worth knowing: only i is parallelized here, not k.
 * That means each row of C is still summed in exactly the same
 * k=0..n-1 order as the serial version -- so omp.c's floating-point
 * result should match serial.c's bit-for-bit (not just "close to",
 * genuinely identical), since summation order per output element is
 * unchanged. That gives a much stronger sanity check than comparing
 * to the oracle alone: omp.c's checksum should match serial.c's
 * checksum exactly, on top of both being close to the NumPy oracle.
 */

#include "types.h"
#include "genmat.h"
#include "bench.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const real *A;
    const real *B;
    real *C;
    int n;
} matmul_args;

static void matmul_kernel_ikj_omp(const real *A, const real *B, real *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = (real)0.0;
        }
        for (int k = 0; k < n; k++) {
            real a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

static void matmul_wrapper(void *arg) {
    matmul_args *ma = (matmul_args *)arg;
    matmul_kernel_ikj_omp(ma->A, ma->B, ma->C, ma->n);
}

int main(int argc, char **argv) {
    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        fprintf(stderr, "Usage: %s -n <N> [-r <reps>] [-t <threads>]\n", argv[0]);
        fprintf(stderr, "  -n N is required: multiplies two N x N matrices.\n");
        fprintf(stderr, "  -t sets the OpenMP thread count (defaults to 1 if omitted --\n");
        fprintf(stderr, "     pass -t explicitly to actually use multiple cores).\n");
        return 1;
    }

    int n = cfg.n;

    /* omp_set_num_threads must be called BEFORE the parallel region
     * that should use it -- it sets the thread count for all
     * subsequent #pragma omp parallel blocks in this program, not
     * just the next one. Called here, once, up front. */
    omp_set_num_threads(cfg.threads);

    real *A = (real *)malloc((size_t)n * (size_t)n * sizeof(real));
    real *B = (real *)malloc((size_t)n * (size_t)n * sizeof(real));
    real *C = (real *)malloc((size_t)n * (size_t)n * sizeof(real));
    if (!A || !B || !C) {
        fprintf(stderr, "allocation failed for n=%d (n*n*sizeof(real) = %zu bytes per matrix)\n",
                n, (size_t)n * (size_t)n * sizeof(real));
        free(A);
        free(B);
        free(C);
        return 1;
    }

    genmat_random(A, n, n, 42u);
    genmat_random(B, n, n, 43u);

    matmul_args ma = { A, B, C, n };

    double time_min, time_med;
    bench_run(matmul_wrapper, &ma, &cfg, &time_min, &time_med);

    double flops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = flops / time_med / 1e9;

    /* Same working-set-estimate caveat as serial.c -- see that file's
     * comment for the full explanation. Not measured traffic. */
    double bytes = 3.0 * (double)n * (double)n * (double)sizeof(real);
    double gbytes_s = bytes / time_med / 1e9;

    bench_report_csv(stdout, "matmul", "omp", n, cfg.threads,
                      time_min, time_med, gflops, gbytes_s);

    real checksum = (real)0.0;
    for (int i = 0; i < n * n; i++) {
        checksum += C[i];
    }
    fprintf(stderr, "sanity: sum(C) = " REAL_FMT ", C[0][0] = " REAL_FMT ", C[%d][%d] = " REAL_FMT "\n",
            checksum, C[0], n - 1, n - 1, C[(size_t)(n - 1) * n + (n - 1)]);

    free(A);
    free(B);
    free(C);
    return 0;
}
