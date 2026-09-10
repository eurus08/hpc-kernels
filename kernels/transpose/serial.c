/*
 * kernels/transpose/serial.c
 *
 * Serial matrix transpose: B = A^T, out-of-place, for an N x N matrix A.
 * B[j][i] = A[i][j] for every i, j.
 *
 * OUT-OF-PLACE, NOT IN-PLACE (confirmed with Prince before writing this):
 * an in-place upper-triangle swap (swap A[i][j] and A[j][i] for i<j) would
 * save memory, but every other kernel in this repo writes to a separate
 * output buffer, and in-place also complicates the upcoming OpenMP version
 * slightly (two threads must never touch the same swap pair). Writing to
 * a separate B keeps this kernel consistent with reduction/matmul/matvec/
 * gaussian's pattern, and keeps the OpenMP version a one-line change
 * (just parallelize the outer loop -- no shared-pair race to reason
 * about).
 *
 * THE ACTUAL LESSON THIS KERNEL TEACHES: reading A row-by-row (as i
 * increases, A[i][j] for fixed j... no, wait -- as j increases with i
 * fixed, A[i*n+j] walks forward one element at a time: perfectly
 * sequential, cache-friendly) is NOT the same as writing B the same way.
 * With the loop order below (outer i, inner j), each write goes to
 * B[j*n+i] -- as j increases, that address jumps forward by n elements
 * every time, not by 1. So this naive version has a good read pattern
 * (A) and a bad write pattern (B) at the same time; there's no way to
 * make both sequential simultaneously with a single flat loop, because
 * transposing fundamentally exchanges "which dimension is contiguous."
 * Swapping the loop order (outer j, inner i) just flips WHICH buffer
 * gets the bad pattern -- it doesn't fix anything. Fixing this properly
 * needs cache-blocking / tiling, which is exactly what the CUDA phase's
 * shared-memory tiled transpose (D3 in the build plan) demonstrates.
 * This file is deliberately the un-fixed naive baseline, same "one
 * correct version first, let benchmarking reveal the bottleneck"
 * approach every other kernel here has followed.
 *
 * ZERO FLOPS: transpose does no arithmetic at all, only data movement.
 * The CSV's gflops column is legitimately 0.0 for this kernel -- gbytes_s
 * is the only metric that means anything here, since this is a purely
 * bandwidth-bound kernel (read A once, write B once, nothing else).
 *
 * WHY THE SANITY CHECK ISN'T sum(B) (unlike every other kernel so far):
 * transpose only rearranges which values live where -- it never changes
 * WHICH values exist. sum(B) always equals sum(A) even for a completely
 * wrong implementation (e.g. one that accidentally copied A into B
 * unchanged, or transposed only half the matrix) -- a sum checksum can't
 * catch an indexing bug here at all. Following reference/check_transpose.py's
 * own approach instead: print one specific off-diagonal pair, B[0][1] and
 * A[1][0], which must match exactly if the transpose is correct (and
 * reference/check_transpose.py explicitly notes this should be an EXACT
 * match, zero relative error, since there's no floating-point arithmetic
 * to introduce rounding -- any mismatch here is a real bug, not FP noise).
 */

#include <stdio.h>
#include <stdlib.h>

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
    (void)cfg.threads; /* unused -- single-core code, same as other *_serial files */

    const int n = cfg.n;

    real *A = malloc((size_t)n * n * sizeof(real));
    real *B = malloc((size_t)n * n * sizeof(real));
    if (!A || !B) {
        fprintf(stderr, "allocation failed for n=%d\n", n);
        free(A); free(B);
        return 1;
    }

    /* Only one input matrix needed -- unlike matmul/matvec/gaussian, there's
     * no second input here, so only the one A/seed-42 convention applies. */
    genmat_random(A, n, n, 42);

    transpose_arg arg = { A, B, n };

    double time_min, time_med;
    bench_run(transpose_kernel, &arg, &cfg, &time_min, &time_med);

    /* B[0][1] should exactly equal A[1][0] -- see the file header for why
     * this is a stronger sanity check here than a sum() would be. Only
     * meaningful for n >= 2. */
    real b01 = (n >= 2) ? B[0 * n + 1] : (real)0.0;
    real a10 = (n >= 2) ? A[1 * n + 0] : (real)0.0;
    fprintf(stderr, "B[0][1]=%.7e A[1][0]=%.7e (should match exactly)\n",
            (double)b01, (double)a10);

    /* No arithmetic in this kernel -- gflops is legitimately 0.0. */
    double gflops = 0.0;
    /* Read A (n^2) once, write B (n^2) once. */
    double gbytes_s = (2.0 * (double)n * (double)n) * sizeof(real) / time_min / 1e9;

    bench_report_csv(stdout, "transpose", "serial", n, 1,
                      time_min, time_med, gflops, gbytes_s);

    free(A);
    free(B);
    return 0;
}
