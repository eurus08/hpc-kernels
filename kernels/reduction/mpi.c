/*
 * kernels/reduction/mpi.c
 *
 * MPI reduction -- same two modes as serial.c/omp.c (-m sum, -m pi), but
 * the shape is fundamentally different from the shared-memory paradigms
 * because MPI processes ("ranks") do NOT share memory, even on one
 * machine. Every rank runs this exact same compiled binary (SPMD --
 * Single Program, Multiple Data) and figures out its own share of the
 * work from its rank number.
 *
 * -m sum needs real communication: to match the same oracle-verified
 * data serial.c/omp.c used, rank 0 generates the FULL array exactly as
 * serial.c does (same seed 42), then MPI_Scatterv splits it into pieces
 * and sends one piece to each rank. This communication cost is real,
 * unavoidable MPI overhead and is deliberately included in the timed
 * region -- unlike serial.c/omp.c, where the data already existed in
 * shared memory before timing started.
 *
 * -m pi needs none: every rank can compute its own slice of sample
 * points directly from its rank number with no data transfer at all
 * (only the final answer needs combining). This makes pi "embarrassingly
 * parallel" under MPI in a way sum is not -- worth noting for write-up.
 *
 * MPI_Reduce combines every rank's local partial sum into one final
 * answer on rank 0 -- the message-passing equivalent of OpenMP's
 * "reduction(+:acc)" clause, but across separate process memory spaces
 * instead of within one shared address space.
 *
 * NOTE: -t is NOT used to control rank count here -- rank count comes
 * from `mpirun -np <N>` at launch time, before this program even starts.
 * -t is left available (default 1) purely so bench_parse_args doesn't
 * need modification, but its value is ignored.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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

/* Everything reduce_kernel needs, bundled for bench_run's void* interface
 * -- same pattern as serial.c/omp.c, but now carrying MPI-specific state
 * (rank, size, the block-distribution tables, and per-rank buffers). */
typedef struct {
    const char *mode;
    int n;
    int rank;
    int size;
    const int *counts; /* counts[r] = how many elements rank r owns */
    const int *displs;  /* displs[r] = starting global index of rank r's slice */
    real *full;          /* valid only on rank 0, sum mode: the whole seed-42 array */
    real *local;         /* every rank's own chunk (scattered-into for sum, computed-into for pi) */
    real result;          /* valid only on rank 0, after MPI_Reduce */
} reduce_arg;

static void reduce_kernel(void *arg_v) {
    reduce_arg *arg = (reduce_arg *)arg_v;
    int my_n = arg->counts[arg->rank];
    real local_sum = (real)0.0;

    /* Align every rank before starting the timed collective work, so
     * one rank arriving late doesn't get silently absorbed into
     * another rank's "compute" time. */
    MPI_Barrier(MPI_COMM_WORLD);

    if (strcmp(arg->mode, "sum") == 0) {
        MPI_Scatterv(arg->full, arg->counts, arg->displs, MPI_REAL_T,
                     arg->local, my_n, MPI_REAL_T,
                     0, MPI_COMM_WORLD);
        for (int i = 0; i < my_n; i++) {
            local_sum += f_identity(arg->local[i]) * (real)1.0;
        }
    } else { /* "pi" */
        int start = arg->displs[arg->rank];
        real dx = (real)1.0 / (real)arg->n;
        for (int i = 0; i < my_n; i++) {
            real x = ((real)(start + i) + (real)0.5) / (real)arg->n;
            local_sum += f_circle(x) * dx;
        }
    }

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
    /* MPI_Init must run before any other MPI call, and conventionally
     * before application argument parsing -- some MPI implementations
     * strip their own reserved arguments out of argv here. */
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *mode = extract_mode(&argc, argv);

    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "usage: mpirun -np <ranks> %s -n <N> [-r <reps>] [-m sum|pi]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    /* Block distribution: split N elements across `size` ranks as evenly
     * as possible. If N doesn't divide evenly, the first (N % size)
     * ranks get one extra element each, rather than dumping the whole
     * remainder on the last rank. Every rank computes this identically
     * and independently -- no communication needed, since every rank
     * already knows N and size. */
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
        /* Same call, same seed, as serial.c/omp.c -- guarantees the
         * scattered data is identical to what the oracle was checked
         * against, not just "some random data". */
        genmat_random(full, cfg.n, 1, 42u);
    }

    real *local = malloc(sizeof(real) * (size_t)counts[rank]);

    reduce_arg arg;
    arg.mode = mode;
    arg.n = cfg.n;
    arg.rank = rank;
    arg.size = size;
    arg.counts = counts;
    arg.displs = displs;
    arg.full = full;
    arg.local = local;
    arg.result = (real)0.0;

    double time_min, time_med;
    bench_run(reduce_kernel, &arg, &cfg, &time_min, &time_med);

    /* Each rank's timer_now() calls wrap its own copy of the same
     * collective calls, so measured times can differ slightly rank to
     * rank. A distributed run is only as fast as its slowest
     * participant, so report the worst case (max) across ranks, not
     * just rank 0's -- standard practice for MPI benchmarking. Done
     * here with an extra MPI_Reduce rather than by modifying
     * common/bench.c, which has no notion of MPI ranks. */
    double local_min = time_min, local_med = time_med;
    double global_min = 0.0, global_med = 0.0;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_med, &global_med, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double flops = 2.0 * (double)cfg.n;
        double bytes = (double)cfg.n * (double)sizeof(real);
        double gflops = (global_min > 0.0) ? (flops / global_min) / 1e9 : 0.0;
        double gbytes_s = (global_min > 0.0) ? (bytes / global_min) / 1e9 : 0.0;

        bench_report_csv(stdout, "reduction", "mpi", cfg.n, size,
                          global_min, global_med, gflops, gbytes_s);

        fprintf(stderr, "mode=%s ranks=%d result=" REAL_FMT "\n", mode, size, arg.result);

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
