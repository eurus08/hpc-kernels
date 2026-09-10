#ifndef HPC_KERNELS_GENMAT_H
#define HPC_KERNELS_GENMAT_H

#include "types.h"

/*
 * Deterministic matrix/vector generators.
 *
 * All three functions are seeded from a plain unsigned int. Same seed
 * always produces the exact same output, on any machine, forever —
 * this is what lets reference/*.py regenerate identical input later
 * for oracle comparison (see genmat.c for the exact PRNG spec that
 * makes that reproducibility promise hold).
 *
 * Matrices are stored row-major, flattened: A[i*cols + j] is row i, col j.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Fills rows*cols entries with uniform values in [-1, 1). Works for
 * matrices (rows, cols both > 1) or vectors (cols == 1). */
void genmat_random(real *A, int rows, int cols, unsigned int seed);

/* Fills an n x n matrix that is strictly diagonally dominant:
 * |A[i][i]| > sum of |A[i][j]| for all j != i, every row.
 * Needed for Gaussian elimination to be numerically stable without
 * pivoting (see genmat.c comment for why). */
void genmat_diag_dominant(real *A, int n, unsigned int seed);

/* Fills an n x n symmetric positive-definite matrix. */
void genmat_spd(real *A, int n, unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif /* HPC_KERNELS_GENMAT_H */
