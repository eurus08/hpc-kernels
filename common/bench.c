#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"

#define BENCH_WARMUP_ITERS 2
#define BENCH_DEFAULT_REPS 5
#define BENCH_DEFAULT_THREADS 1

int bench_parse_args(int argc, char **argv, bench_config *cfg) {
    cfg->n = -1; /* sentinel: "not provided" */
    cfg->reps = BENCH_DEFAULT_REPS;
    cfg->threads = BENCH_DEFAULT_THREADS;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cfg->n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            cfg->reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            cfg->threads = atoi(argv[++i]);
        }
    }

    if (cfg->n <= 0) return 1;       /* -n is required and must be positive */
    if (cfg->reps <= 0) return 1;
    if (cfg->threads <= 0) return 1;

    return 0;
}

/* Ascending comparator for qsort — standard library sort needs a function
 * that says "which of these two comes first," this is that function. */
static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

void bench_run(void (*kernel_fn)(void *arg), void *arg,
               const bench_config *cfg,
               double *time_min, double *time_med) {
    for (int i = 0; i < BENCH_WARMUP_ITERS; ++i) {
        kernel_fn(arg);
    }

    double *samples = (double *)malloc((size_t)cfg->reps * sizeof(double));

    for (int i = 0; i < cfg->reps; ++i) {
        double t0 = timer_now();
        kernel_fn(arg);
        double t1 = timer_now();
        samples[i] = t1 - t0;
    }

    qsort(samples, (size_t)cfg->reps, sizeof(double), compare_double);

    *time_min = samples[0]; /* sorted ascending, so index 0 is the minimum */

    /* Median: middle element for odd counts, average of the two middle
     * elements for even counts. */
    if (cfg->reps % 2 == 1) {
        *time_med = samples[cfg->reps / 2];
    } else {
        *time_med = (samples[cfg->reps / 2 - 1] + samples[cfg->reps / 2]) / 2.0;
    }

    free(samples);
}

void bench_report_csv(FILE *out,
                       const char *kernel, const char *paradigm,
                       int n, int threads,
                       double time_min, double time_med,
                       double gflops, double gbytes_s) {
#ifdef USE_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif

    fprintf(out, "%s,%s,%s,%d,%d,%.9e,%.9e,%.6e,%.6e\n",
            kernel, paradigm, precision, n, threads,
            time_min, time_med, gflops, gbytes_s);
}
