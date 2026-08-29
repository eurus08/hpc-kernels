#ifndef HPC_KERNELS_BENCH_H
#define HPC_KERNELS_BENCH_H

#include "types.h"
#include <stdio.h>  /* for FILE *, used in bench_report_csv */

/*
 * Uniform benchmarking harness.
 *
 * Every benchmark binary (bin/matmul_omp -n 2048 -r 5 -t 8, etc.) parses
 * the same three flags, runs the same warm-up + timing discipline, and
 * emits the same CSV shape. This file is what makes that uniformity
 * automatic instead of copy-pasted eleven times with eleven chances to
 * drift out of sync.
 */

typedef struct {
    int n;        /* problem size, from -n (required) */
    int reps;     /* timed repetitions, from -r (default 5) */
    int threads;  /* thread count or MPI rank count, from -t (default 1) */
} bench_config;

/* Parses -n <int> -r <int> -t <int> from argv. -n is required; -r and -t
 * fall back to their defaults (5, 1) if omitted. Returns 0 on success,
 * nonzero if -n was missing or a value failed to parse — caller should
 * print usage and exit in that case. */
int bench_parse_args(int argc, char **argv, bench_config *cfg);

/*
 * Runs kernel_fn(arg) repeatedly: 2 warm-up calls (discarded, per the
 * plan's spec — first runs are artificially slow from cold caches and
 * settling driver/OS state), then cfg->reps timed calls.
 *
 * Writes the minimum and median elapsed time (seconds) into *time_min
 * and *time_med. Median, not mean: a single scheduler hiccup can spike
 * one repetition badly, and a mean lets that one bad sample drag the
 * whole result; the median just ignores it.
 */
void bench_run(void (*kernel_fn)(void *arg), void *arg,
                const bench_config *cfg,
                double *time_min, double *time_med);

/*
 * Emits one CSV line to `out` in the fixed column order:
 * kernel,paradigm,precision,N,threads_or_ranks,time_min,time_med,gflops,gbytes_s
 *
 * precision is filled in automatically from REAL_FMT (types.h) as
 * "float" or "double" — callers never pass it explicitly, so there's
 * no way for a benchmark binary to accidentally mislabel which
 * precision it was actually compiled with.
 */
void bench_report_csv(FILE *out,
                       const char *kernel, const char *paradigm,
                       int n, int threads,
                       double time_min, double time_med,
                       double gflops, double gbytes_s);

#endif /* HPC_KERNELS_BENCH_H */
