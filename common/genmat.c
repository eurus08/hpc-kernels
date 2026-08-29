#include "genmat.h"
#include <stdint.h>
#include <math.h>

/*
 * xorshift32 — George Marsaglia's minimal PRNG.
 *
 * Chosen deliberately over libc's rand(): rand()'s actual algorithm is
 * left implementation-defined by the C standard, so the same seed can
 * give different sequences on glibc vs musl vs anywhere else, and
 * NumPy's generator is a different algorithm again. That silently
 * breaks the "Python oracle regenerates identical input" requirement.
 * xorshift32 is three lines of fully-specified integer arithmetic —
 * trivial to reimplement bit-for-bit in Python's reference/ scripts.
 */
static uint32_t xorshift32_step(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Raw 32-bit output -> real in [-1, 1). */
static real rand_uniform(uint32_t *state) {
    uint32_t r = xorshift32_step(state);
    double u = (double)r / 4294967296.0; /* 2^32 */
    return (real)(2.0 * u - 1.0);
}

void genmat_random(real *A, int rows, int cols, unsigned int seed) {
    uint32_t state = seed ? seed : 1u; /* xorshift32 requires a nonzero seed */
    for (int i = 0; i < rows * cols; ++i) {
        A[i] = rand_uniform(&state);
    }
}

void genmat_diag_dominant(real *A, int n, unsigned int seed) {
    uint32_t state = seed ? seed : 1u;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) A[i * n + j] = rand_uniform(&state);
        }
    }
    /* Set each diagonal entry to strictly exceed its row's off-diagonal
     * absolute sum. This is the literal definition of diagonal dominance. */
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) row_sum += fabs((double)A[i * n + j]);
        }
        A[i * n + i] = (real)(row_sum + 1.0);
    }
}

void genmat_spd(real *A, int n, unsigned int seed) {
    uint32_t state = seed ? seed : 1u;

    /* Fill upper triangle, mirror into lower triangle -> symmetric by construction. */
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            real v = rand_uniform(&state);
            A[i * n + j] = v;
            A[j * n + i] = v;
        }
    }
    /* Diagonal: strictly diagonally dominant + symmetric + positive diagonal
     * guarantees positive-definite, via the Gershgorin circle theorem
     * (every eigenvalue lies within row_sum of the diagonal entry, so a
     * diagonal large enough relative to its row forces all eigenvalues
     * positive). This avoids an O(n^3) R^T*R multiply just to build a
     * test matrix — O(n^2) instead, which matters once n reaches 4096. */
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) row_sum += fabs((double)A[i * n + j]);
        }
        A[i * n + i] = (real)(row_sum + (double)n);
    }
}
