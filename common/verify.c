#include "verify.h"
#include <stdio.h>
#include <math.h>

int verify_result(const real *computed, const real *expected, int n, real tol) {
    double diff_sq_sum = 0.0;   /* sum of (computed - expected)^2, for 2-norm */
    double exp_sq_sum  = 0.0;   /* sum of expected^2, to normalize the error  */
    double max_abs_diff = 0.0;

    for (int i = 0; i < n; ++i) {
        double d = (double)computed[i] - (double)expected[i];
        double e = (double)expected[i];

        diff_sq_sum += d * d;
        exp_sq_sum  += e * e;

        double abs_d = fabs(d);
        if (abs_d > max_abs_diff) max_abs_diff = abs_d;
    }

    double diff_norm = sqrt(diff_sq_sum);
    double exp_norm   = sqrt(exp_sq_sum);

    /* Guard against dividing by ~zero when the expected vector/matrix
     * is itself all zeros (or vanishingly small) — fall back to the
     * raw (unnormalized) diff norm in that edge case rather than
     * producing inf/nan and calling it a "relative error." */
    double rel_error = (exp_norm > 1e-300) ? (diff_norm / exp_norm) : diff_norm;

    int passed = (rel_error <= (double)tol);

    printf("verify: relative_error(2-norm)=%.6e  max_abs_diff=%.6e  tol=%.6e  -> %s\n",
           rel_error, max_abs_diff, (double)tol, passed ? "PASS" : "FAIL");

    return passed;
}
