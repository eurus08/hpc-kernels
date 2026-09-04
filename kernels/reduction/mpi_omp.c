/*
 * kernels/reduction/mpi_omp.c
 *
 * Hybrid MPI+OpenMP reduction: MPI still splits the DATA across ranks
 * (identical block distribution to mpi.c/mpi_subcomm.c), but each rank
 * now processes its own local chunk with multiple OpenMP threads
 * instead of a single-threaded loop. Total workers = ranks * threads
 * -- e.g. 2 MPI ranks x 3 OpenMP threads each = 6 total workers on a
 * 6-core machine, instead of 6 flat MPI ranks (mpi.c) or 6 flat
 * OpenMP threads (omp.c). This is the pattern most production HPC
 * codes actually use on multi-core cluster nodes.
 *
 * -t now means "OpenMP threads per rank" (not MPI rank count -- that
 * still comes from `mpirun -np`, same convention as mpi.c/
 * mpi_subcomm.c).
 *
 * MPI_Init_thread (not plain MPI_Init) is required once OpenMP threads
 * are in the picture: it negotiates a THREAD SUPPORT LEVEL with the
 * MPI implementation. We request MPI_THREAD_FUNNELED, meaning "only
 * the thread that called MPI_Init_thread will ever make MPI calls" --
 * true here, since every MPI_Scatterv/MPI_Reduce/MPI_Barrier call
 * happens on the main thread, outside any OpenMP parallel region; the
 * OpenMP threads only ever do local arithmetic, never touch MPI. We
 * check MPI actually granted at least that level, since an
 * implementation may silently give a lower level otherwise.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "types.h"
#include "timer.h"
#include "genmat.h"
#include "verify.h"
#include "bench.h"

#ifdef USE_DOUBLE
#define MPI_REAL_T MPI_DOUBLE
#else
#define MPI_REAL_T MPI_FLOAT
#endif

typedef real (*integrand_fn)(real x);

static real f_identity(real x) {
    return x;
}

static real f_circle(real x) {
    return (real)4.0 / ((real)1.0 + x * x);
}

typedef struct {
    const char *mode;
    int n;
    int rank;
    const int *counts;
    const int *displs;
    real *full;   /* valid only on rank 0, sum mode */
    real *local;  /* this rank's own chunk */
    real result;  /* valid only on rank 0, after MPI_Reduce */
} reduce_arg;

static void reduce_kernel(void *arg_v) {
    reduce_arg *arg = (reduce_arg *)arg_v;
    int my_n = arg->counts[arg->rank];
    real local_sum = (real)0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    if (strcmp(arg->mode, "sum") == 0) {
        /* MPI moves the data to this rank -- same Scatterv as mpi.c. */
        MPI_Scatterv(arg->full, arg->counts, arg->displs, MPI_REAL_T,
                     arg->local, my_n, MPI_REAL_T,
                     0, MPI_COMM_WORLD);

        /* OpenMP now parallelizes THIS rank's own local loop over its
         * chunk -- exactly the same reduction clause omp.c used, just
         * operating on a smaller, rank-local slice instead of the
         * whole array. */
        const real *local_data = arg->local;
        #pragma omp parallel for reduction(+:local_sum)
        for (int i = 0; i < my_n; i++) {
            local_sum += f_identity(local_data[i]) * (real)1.0;
        }
    } else { /* "pi" */
        int start = arg->displs[arg->rank];
        int n_total = arg->n;
        real dx = (real)1.0 / (real)n_total;

        #pragma omp parallel for reduction(+:local_sum)
        for (int i = 0; i < my_n; i++) {
            real x = ((real)(start + i) + (real)0.5) / (real)n_total;
            local_sum += f_circle(x) * dx;
        }
    }

    /* Combine across ranks -- flat MPI_Reduce, same as mpi.c. The
     * hierarchical (mpi_subcomm.c) and hybrid (this file) refinements
     * are independent of each other; this file keeps the combine step
     * simple so the only new variable is "threads per rank". */
    real global_sum = (real)0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_REAL_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (arg->rank == 0) {
        arg->result = global_sum;
    }
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED && rank == 0) {
        fprintf(stderr,
                "warning: MPI implementation granted a lower thread support "
                "level (%d) than MPI_THREAD_FUNNELED (%d) requested -- hybrid "
                "MPI+OpenMP calls may not be safe on this system.\n",
                provided, MPI_THREAD_FUNNELED);
    }

    const char *mode = extract_mode(&argc, argv);

    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "usage: mpirun -np <ranks> %s -n <N> [-r <reps>] [-t <threads_per_rank>] [-m sum|pi]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    /* -t means OpenMP threads PER RANK here, set once before the
     * timed loop -- same idea as omp.c, just scoped to one rank. */
    omp_set_num_threads(cfg.threads);

    int *counts = malloc(sizeof(int) * (size_t)size);
    int *displs = malloc(sizeof(int) * (size_t)size);
    int base = cfg.n / size;
    int rem = cfg.n % size;
    int offset = 0;
    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    real *full = NULL;
    if (rank == 0 && strcmp(mode, "sum") == 0) {
        full = malloc(sizeof(real) * (size_t)cfg.n);
        genmat_random(full, cfg.n, 1, 42u);
    }
    real *local = malloc(sizeof(real) * (size_t)counts[rank]);

    reduce_arg arg;
    arg.mode = mode;
    arg.n = cfg.n;
    arg.rank = rank;
    arg.counts = counts;
    arg.displs = displs;
    arg.full = full;
    arg.local = local;
    arg.result = (real)0.0;

    double time_min, time_med;
    bench_run(reduce_kernel, &arg, &cfg, &time_min, &time_med);

    double local_min = time_min, local_med = time_med;
    double global_min = 0.0, global_med = 0.0;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_med, &global_med, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double flops = 2.0 * (double)cfg.n;
        double bytes = (double)cfg.n * (double)sizeof(real);
        double gflops = (global_min > 0.0) ? (flops / global_min) / 1e9 : 0.0;
        double gbytes_s = (global_min > 0.0) ? (bytes / global_min) / 1e9 : 0.0;

        /* CSV "threads_or_ranks" column reports MPI world size, same
         * convention as mpi.c/mpi_subcomm.c; total worker count
         * (size * cfg.threads) is reported separately on stderr,
         * since bench_report_csv's schema is fixed and shouldn't be
         * changed just for this one paradigm. */
        bench_report_csv(stdout, "reduction", "mpi_omp", cfg.n, size,
                          global_min, global_med, gflops, gbytes_s);

        fprintf(stderr,
                "mode=%s ranks=%d threads_per_rank=%d total_workers=%d result=" REAL_FMT "\n",
                mode, size, cfg.threads, size * cfg.threads, arg.result);

        real tol = (real)-1.0;
        real expected = (real)0.0;
        if (strcmp(mode, "pi") == 0) {
            expected = (real)M_PI;
            tol = (real)1e-2;
        }

        if (tol >= (real)0.0) {
            real computed_arr[1] = { arg.result };
            real expected_arr[1] = { expected };
            verify_result(computed_arr, expected_arr, 1, tol);
        } else {
            fprintf(stderr,
                    "note: 'sum' mode is not verified inline (no closed-form "
                    "reference); compare against reference/check_reduction.py.\n");
        }
    }

    free(local);
    free(counts);
    free(displs);
    if (full != NULL) {
        free(full);
    }

    MPI_Finalize();
    return 0;
}
