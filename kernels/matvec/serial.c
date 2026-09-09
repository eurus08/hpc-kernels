/*
 * kernels/matvec/serial.c
 *
 * Serial (single-core) dense matrix-vector multiply: y = A * x, for an
 * N x N matrix A and length-N vectors x, y.
 *
 * Arithmetic intensity note (why there's no loop-reordering trick here,
 * unlike matmul/serial.c's ikj ordering):
 *
 *   Matmul does O(n^3) floating-point work on O(n^2) of data -- each
 *   element of A and B gets reused n times, so *how* you order the
 *   loops changes how much of that reuse actually lands in cache.
 *
 *   Matvec does only O(n^2) work on O(n^2) of data -- each element of
 *   A is read exactly once, multiplied by one element of x, and added
 *   in. There's no reuse to exploit by reordering. The natural
 *   row-by-row dot product below already walks both A's row and x
 *   sequentially (the cache-friendly pattern), so it's also the
 *   fastest pattern -- nothing to hoist or reorder.
 *
 *   Net effect: matvec is *bandwidth-bound* (like reduction was),
 *   not *compute-bound* (like matmul was). Roughly 2 flops per real
 *   read from memory, so moving the data dominates, not the math.
 */

#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "genmat.h"
#include "bench.h"

/* Bundles everything the timed kernel function needs into one struct,
 * since bench_run() calls kernel_fn(void *arg) with a single opaque
 * pointer -- this is how we hand it A, x, y, and n without globals. */
typedef struct {
    const real *A;
    const real *x;
    real *y;
    int n;
} matvec_arg;

/* y = A * x, one dot product per row of A. */
static void matvec_kernel(void *arg)
{
    matvec_arg *a = (matvec_arg *)arg;
    const int n = a->n;
    const real *A = a->A;
    const real *x = a->x;
    real *y = a->y;

    for (int i = 0; i < n; i++) {
        const real *row = A + (size_t)i * n;  /* row i, contiguous in memory */
        real sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
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
    (void)cfg.threads; /* accepted for CLI uniformity with other paradigms;
                           unused -- this is single-core code, same as
                           matmul/serial.c's -t handling. */

    const int n = cfg.n;

    real *A = malloc((size_t)n * n * sizeof(real));
    real *x = malloc((size_t)n * sizeof(real));
    real *y = malloc((size_t)n * sizeof(real));
    if (!A || !x || !y) {
        fprintf(stderr, "allocation failed for n=%d\n", n);
        free(A); free(x); free(y);
        return 1;
    }

    /* Same seeds as matmul's A/B convention (42, 43), so
     * reference/check_matvec.py can regenerate identical A and x. */
    genmat_random(A, n, n, 42);
    genmat_random(x, n, 1, 43);

    matvec_arg arg = { A, x, y, n };

    double time_min, time_med;
    bench_run(matvec_kernel, &arg, &cfg, &time_min, &time_med);

    /* Sanity line for eyeball/external-oracle comparison, same pattern
     * as matmul/serial.c -- check_matvec.py doesn't print a directly
     * comparable value on its own, so this gives us something to diff
     * against expected_matvec.npy by hand. */
    real sum_y = 0.0;
    for (int i = 0; i < n; i++) sum_y += y[i];
    fprintf(stderr, "sum(y)=%.7e y[0]=%.7e y[n-1]=%.7e\n",
            (double)sum_y, (double)y[0], (double)y[n - 1]);

    /* 2*n^2 flops total: n multiplies + n adds per row, n rows. */
    double flops = 2.0 * (double)n * (double)n;
    double gflops = flops / time_min / 1e9;

    /* Working-set estimate, same caveat as every other kernel's CSV:
     * read A (n^2) once, read x (n) once, write y (n) once -- ignores
     * that x is re-read from cache on every row, and ignores any real
     * cache-line-level traffic. Revisited properly at Phase 5. */
    double gbytes_s = ((double)n * n + 2.0 * n) * sizeof(real) / time_min / 1e9;

    bench_report_csv(stdout, "matvec", "serial", n, 1,
                      time_min, time_med, gflops, gbytes_s);

    free(A);
    free(x);
    free(y);
    return 0;
}
