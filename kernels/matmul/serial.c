/*
 * kernels/matmul/serial.c
 *
 * Serial (single-core) dense matrix-matrix multiply: C = A * B, for
 * square N x N matrices, using the standard "triple loop" algorithm --
 * no blocking/tiling, no vectorization tricks. This is the baseline
 * everything else (OpenMP, MPI/Cannon's) gets compared against.
 *
 * Matrices are generated with common/genmat.h's genmat_random, seeded
 * 42 for A and 43 for B -- these exact seeds match reference/check_matmul.py
 * (`A = gen(N, seed=42); B = gen(N, seed=43)`), so the C program and the
 * Python oracle are multiplying the identical two matrices.
 *
 * Storage: row-major, flattened, per common/genmat.h's convention --
 * A[i][j] lives at A[i*n + j].
 */

#include "types.h"
#include "genmat.h"
#include "bench.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * bench_run's signature is void (*kernel_fn)(void *arg) -- it can only
 * hand the kernel one pointer. This struct bundles everything
 * matmul_wrapper needs (the three matrices and the dimension) into
 * that one pointer.
 */
typedef struct {
    const real *A;
    const real *B;
    real *C;
    int n;
} matmul_args;

/*
 * Core computation: C = A * B for n x n row-major matrices.
 *
 * Loop order is i -> k -> j, not the "obvious" i -> j -> k. Both
 * orderings compute exactly the same O(n^3) sum, in a different
 * grouping, but they access memory very differently:
 *
 *   i -> j -> k (naive):
 *     for i: for j: for k: C[i][j] += A[i][k] * B[k][j]
 *   Here B[k][j] is read with k varying in the INNERMOST loop, which
 *   means jumping n elements through memory on every single step
 *   (walking down a column of a row-major matrix). Every step likely
 *   needs a fresh trip to memory.
 *
 *   i -> k -> j (this file):
 *     for i: for k: for j: C[i][j] += A[i][k] * B[k][j]
 *   Here A[i][k] is hoisted out as a scalar (a_ik) before the
 *   innermost loop, and both B[k][j] and C[i][j] are walked with j
 *   varying in the innermost loop -- i.e. straight along a row, one
 *   element after another. Since matrices are stored row-major (a
 *   whole row is contiguous in memory) and the CPU fetches memory in
 *   fixed-size chunks called cache lines (typically 64 bytes -- enough
 *   for several `real`s at once), walking along a row means each
 *   fetched chunk gets reused for several loop iterations instead of
 *   being fetched once and thrown away. That reuse is the entire
 *   difference in speed -- same math, same result, far fewer trips to
 *   memory.
 *
 * This is a purely mechanical optimization (loop reordering), not an
 * algorithmic one -- still O(n^3), still the textbook algorithm.
 * Blocking/tiling (revisiting cache reuse at a coarser granularity)
 * is a further optimization deliberately left out of this baseline
 * file.
 */
static void matmul_kernel_ikj(const real *A, const real *B, real *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = (real)0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            real a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

/* Wrapper matching bench_run's required void(*)(void*) shape. */
static void matmul_wrapper(void *arg) {
    matmul_args *ma = (matmul_args *)arg;
    matmul_kernel_ikj(ma->A, ma->B, ma->C, ma->n);
}

int main(int argc, char **argv) {
    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        fprintf(stderr, "Usage: %s -n <N> [-r <reps>] [-t <threads>]\n", argv[0]);
        fprintf(stderr, "  -n N is required: multiplies two N x N matrices.\n");
        fprintf(stderr, "  -t is accepted for CLI uniformity but unused here --\n");
        fprintf(stderr, "     this is single-core serial code.\n");
        return 1;
    }

    int n = cfg.n;

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

    /* 2*n^3 floating-point operations: n^3 multiplications + n^3
     * additions, one of each per (i,j,k) triple. */
    double flops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = flops / time_med / 1e9;

    /* NOTE (open design question, not yet resolved -- flagging rather
     * than deciding silently): this is a WORKING-SET estimate, not
     * measured memory traffic -- reading all of A and B once plus
     * writing all of C once, in bytes. It assumes perfect reuse (every
     * byte fetched from RAM exactly once), which the naive triple loop
     * does NOT actually achieve once n is large enough that B and C
     * stop fitting in cache -- real DRAM traffic will be higher than
     * this figure. Matmul is compute-bound at any reasonable N (unlike
     * reduction, which was bandwidth-bound), so gflops is the metric
     * that actually matters here; this gbytes_s figure is a rough
     * lower bound, not ground truth. Worth revisiting properly at
     * Phase 5's roofline stage rather than treating it as measured. */
    double bytes = 3.0 * (double)n * (double)n * (double)sizeof(real);
    double gbytes_s = bytes / time_med / 1e9;

    bench_report_csv(stdout, "matmul", "serial", n, cfg.threads,
                      time_min, time_med, gflops, gbytes_s);

    /* Cheap sanity signal for manual cross-checking against the
     * Python oracle while we sort out a real automated comparison
     * (see note below) -- sum of all of C's elements plus the two
     * corner elements. Not a substitute for real verification. */
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
