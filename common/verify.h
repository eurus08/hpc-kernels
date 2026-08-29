#ifndef HPC_KERNELS_VERIFY_H
#define HPC_KERNELS_VERIFY_H

#include "types.h"

/*
 * Correctness checking against a reference (oracle) result.
 *
 * Reports BOTH relative error (2-norm) and max absolute difference —
 * never collapse this to a bare PASS/FAIL. A bare pass/fail can't
 * distinguish "off by float rounding noise" from "actually broken,"
 * and a badly-chosen tolerance can silently hide the difference.
 * Always print the real numbers so a human can judge for themselves.
 *
 * Returns 1 if relative error <= tol, 0 otherwise. Prints either way.
 */
int verify_result(const real *computed, const real *expected, int n, real tol);

#endif /* HPC_KERNELS_VERIFY_H */
