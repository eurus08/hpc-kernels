/*
 * kernels/reduction/serial.c
 *
 * Serial reduction, single core, one loop, no OpenMP/MPI.
 *
 * "Reduction" here covers two modes, not two algorithms:
 *   -m sum   plain summation of N random values           (checked against reference/check_reduction.py)
 *   -m pi    estimate pi via trapezoidal integration of    (checked against the analytic constant M_PI)
 *            4/(1+x^2) over [0,1]
 *
 * Both are the same "map an integrand over a set of points, then add
 * the results up" pattern -- what differs is which points and which
 * function is applied first. That's expressed here with a C function
 * pointer (integrand_fn) instead of writing two near-duplicate loops.
 * See the accompanying chat explanation for what a function pointer is,
 * if this is new.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "types.h"
#include "timer.h"
#include "genmat.h"
#include "verify.h"
#include "bench.h"

/* A function pointer type: "a function taking one real, returning one
 * real". Any function matching that shape can be passed around and
 * called through a variable of this type -- that's what lets one loop
 * below serve both "sum" and "pi" without duplicating it. */
typedef real (*integrand_fn)(real x);

/* mode: sum -- apply nothing, just add the raw values up */
static real f_identity(real x) {
    return x;
}

/* mode: pi -- the function whose area under the curve on [0,1] is pi;
 * see the trapezoidal-rule note below for why this converges to pi */
static real f_circle(real x) {
    return (real)4.0 / ((real)1.0 + x * x);
}

/* Everything the timed kernel function needs, bundled into one struct.
 * bench_run() only knows how to call "void kernel_fn(void *arg)" -- it
 * has no idea what a reduction is -- so all our real arguments (the
 * data, which integrand, the result) have to travel through this one
 * opaque pointer. This is a common C pattern for passing multiple
 * arguments through an interface that only accepts one. */
typedef struct {
    const real *points;
    int n;
    integrand_fn f;
    real dx;
    real result; /* kernel writes its answer back here */
} reduce_arg;

/* The actual timed work. Matches bench_run's required signature exactly:
 * void (*)(void *arg). We immediately cast arg back to the real type. */
static void reduce_kernel(void *arg_v) {
    reduce_arg *arg = (reduce_arg *)arg_v;
    real acc = (real)0.0;
    for (int i = 0; i < arg->n; i++) {
        acc += arg->f(arg->points[i]) * arg->dx;
    }
    arg->result = acc;
}

/*
 * bench_parse_args() only knows about -n / -r / -t (see bench.h) -- it
 * has no idea this binary also needs a -m <sum|pi> flag. Rather than
 * guess whether bench_parse_args tolerates or rejects flags it doesn't
 * recognise, we pull "-m <value>" out of argv ourselves first, closing
 * the gap it leaves behind, so bench_parse_args only ever sees the
 * three flags it actually knows about. argc is updated in place.
 */
static const char *extract_mode(int *argc, char **argv) {
    static char mode[16] = "sum"; /* default if -m is not given */
    for (int i = 1; i < *argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < *argc) {
            strncpy(mode, argv[i + 1], sizeof(mode) - 1);
            mode[sizeof(mode) - 1] = '\0';
            /* shift everything after "-m <value>" left by two, then
             * shrink argc -- this is what "removing" two entries from
             * a C array looks like, since arrays can't shrink in place */
            for (int j = i; j + 2 < *argc; j++) {
                argv[j] = argv[j + 2];
            }
            *argc -= 2;
            break;
        }
    }
    return mode;
}

int main(int argc, char **argv) {
    const char *mode = extract_mode(&argc, argv);

    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        fprintf(stderr,
                "usage: %s -n <N> [-r <reps>] [-t <threads>] [-m sum|pi]\n",
                argv[0]);
        return 1;
    }

    real *points = malloc(sizeof(real) * (size_t)cfg.n);
    if (points == NULL) {
        fprintf(stderr, "malloc failed for N=%d\n", cfg.n);
        return 1;
    }

    integrand_fn f;
    real dx;
    real expected = (real)0.0;
    real tol = (real)-1.0; /* negative = "skip inline verification" */

    if (strcmp(mode, "pi") == 0) {
        /* Trapezoidal rule: split [0,1] into N slices, sample the
         * midpoint of each slice, multiply by the slice width dx, add
         * them up. That approximates the area under f_circle, which
         * equals pi by construction (this is the standard "Monte
         * Carlo-free" way to estimate pi via integration). */
        for (int i = 0; i < cfg.n; i++) {
            points[i] = ((real)i + (real)0.5) / (real)cfg.n;
        }
        f = f_circle;
        dx = (real)1.0 / (real)cfg.n;
        expected = (real)M_PI;
        tol = (real)1e-2; /* trapezoidal convergence is O(1/N); loose on purpose */
    } else if (strcmp(mode, "sum") == 0) {
        genmat_random(points, cfg.n, 1, 42u); /* seed fixed: matches reference/check_reduction.py */
        f = f_identity;
        dx = (real)1.0;
        /* No analytic expected value for a sum of random numbers --
         * verified against reference/check_reduction.py separately,
         * not inline here. */
    } else {
        fprintf(stderr, "unknown mode '%s' (expected 'sum' or 'pi')\n", mode);
        free(points);
        return 1;
    }

    reduce_arg arg;
    arg.points = points;
    arg.n = cfg.n;
    arg.f = f;
    arg.dx = dx;
    arg.result = (real)0.0;

    double time_min, time_med;
    bench_run(reduce_kernel, &arg, &cfg, &time_min, &time_med);

    /* Rough FLOP/byte accounting for the CSV: one multiply-add per
     * element, one real read per element. f()'s own cost (e.g. the
     * divide in f_circle) is not separately counted -- fine for now,
     * this is a coarse bandwidth/compute estimate, not a precise one. */
    double flops = 2.0 * (double)cfg.n;
    double bytes = (double)cfg.n * (double)sizeof(real);
    double gflops = (time_min > 0.0) ? (flops / time_min) / 1e9 : 0.0;
    double gbytes_s = (time_min > 0.0) ? (bytes / time_min) / 1e9 : 0.0;

    bench_report_csv(stdout, "reduction", "serial", cfg.n, cfg.threads,
                      time_min, time_med, gflops, gbytes_s);

    fprintf(stderr, "mode=%s result=" REAL_FMT "\n", mode, arg.result);

    if (tol >= (real)0.0) {
        real computed_arr[1] = { arg.result };
        real expected_arr[1] = { expected };
        verify_result(computed_arr, expected_arr, 1, tol);
    } else {
        fprintf(stderr,
                "note: 'sum' mode is not verified inline (no closed-form "
                "reference); compare against reference/check_reduction.py.\n");
    }

    free(points);
    return 0;
}