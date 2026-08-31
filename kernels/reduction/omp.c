/*
 * kernels/reduction/omp.c
 *
 * OpenMP reduction -- same two modes as serial.c (-m sum, -m pi), same
 * function-pointer integrand design, but the accumulation loop is now
 * split across CPU cores with "#pragma omp parallel for reduction(+:acc)".
 *
 * Why the reduction clause specifically: if every thread wrote directly
 * to one shared `acc` variable, two threads could read the same old
 * value of `acc` before either writes back, and one thread's update
 * would silently vanish (a "race condition" -- the result depends on
 * unpredictable thread timing, not just the input data). The reduction
 * clause avoids this by giving each thread its own private `acc`,
 * accumulating with zero sharing during the loop, and combining the
 * private copies safely only once, after the loop ends.
 *
 * Note: because the private partial sums get combined in a different
 * order than serial.c's straight left-to-right loop, and floating-point
 * addition is not associative ((a+b)+c can differ in its last bit from
 * a+(b+c)), the `sum` result here may not be bit-for-bit identical to
 * serial.c's. That's expected rounding noise, not a bug -- it's exactly
 * why verify_result checks relative error against a tolerance rather
 * than exact equality.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "types.h"
#include "timer.h"
#include "genmat.h"
#include "verify.h"
#include "bench.h"

typedef real (*integrand_fn)(real x);

static real f_identity(real x) {
    return x;
}

static real f_circle(real x) {
    return (real)4.0 / ((real)1.0 + x * x);
}

typedef struct {
    const real *points;
    int n;
    integrand_fn f;
    real dx;
    real result;
} reduce_arg;

/* Same signature bench_run requires: void (*)(void *arg). Only the body
 * changed from serial.c -- one pragma line above the loop. */
static void reduce_kernel(void *arg_v) {
    reduce_arg *arg = (reduce_arg *)arg_v;
    real acc = (real)0.0;
    int n = arg->n;
    integrand_fn f = arg->f;
    real dx = arg->dx;
    const real *points = arg->points;

    #pragma omp parallel for reduction(+:acc)
    for (int i = 0; i < n; i++) {
        acc += f(points[i]) * dx;
    }

    arg->result = acc;
}

static const char *extract_mode(int *argc, char **argv) {
    static char mode[16] = "sum";
    for (int i = 1; i < *argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < *argc) {
            strncpy(mode, argv[i + 1], sizeof(mode) - 1);
            mode[sizeof(mode) - 1] = '\0';
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

    /* -t controls OpenMP thread count here (it meant MPI rank count in
     * serial.c only in the sense that serial.c ignored it -- there was
     * no parallelism to control). Setting it once before bench_run is
     * enough: thread count doesn't change between the timed repetitions. */
    omp_set_num_threads(cfg.threads);

    real *points = malloc(sizeof(real) * (size_t)cfg.n);
    if (points == NULL) {
        fprintf(stderr, "malloc failed for N=%d\n", cfg.n);
        return 1;
    }

    integrand_fn f;
    real dx;
    real expected = (real)0.0;
    real tol = (real)-1.0;

    if (strcmp(mode, "pi") == 0) {
        for (int i = 0; i < cfg.n; i++) {
            points[i] = ((real)i + (real)0.5) / (real)cfg.n;
        }
        f = f_circle;
        dx = (real)1.0 / (real)cfg.n;
        expected = (real)M_PI;
        tol = (real)1e-2;
    } else if (strcmp(mode, "sum") == 0) {
        genmat_random(points, cfg.n, 1, 42u);
        f = f_identity;
        dx = (real)1.0;
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

    double flops = 2.0 * (double)cfg.n;
    double bytes = (double)cfg.n * (double)sizeof(real);
    double gflops = (time_min > 0.0) ? (flops / time_min) / 1e9 : 0.0;
    double gbytes_s = (time_min > 0.0) ? (bytes / time_min) / 1e9 : 0.0;

    bench_report_csv(stdout, "reduction", "omp", cfg.n, cfg.threads,
                      time_min, time_med, gflops, gbytes_s);

    fprintf(stderr, "mode=%s threads=%d result=" REAL_FMT "\n", mode, cfg.threads, arg.result);

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
