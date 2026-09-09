/*
 * kernels/matvec/mpi.c
 *
 * MPI dense matrix-vector multiply: y = A * x, N x N matrix, row-distributed
 * across ranks.
 *
 * DECOMPOSITION (confirmed with Prince before writing this):
 *
 *   Row block sizes are balanced as evenly as possible, not just "give the
 *   remainder to the last rank": with n rows and p ranks, let
 *
 *       base  = n / p     (integer division)
 *       extra = n % p
 *
 *   The first `extra` ranks get (base + 1) rows each; the remaining ranks
 *   get `base` rows each. No rank differs from any other by more than one
 *   row. This also degrades gracefully if p > n: ranks beyond the first n
 *   simply get 0 rows and sit idle for the compute -- no crash, no special
 *   case needed, the same arithmetic just naturally gives them nothing to
 *   do.
 *
 *   Rank r's row range is [row_start(r), row_start(r) + local_rows(r)).
 *
 * WHY x NEEDS NO COMMUNICATION:
 *
 *   genmat_random() is a deterministic PRNG seeded from a plain integer --
 *   same seed always produces the exact same output, on any process, with
 *   no dependency on what else that process has done. So instead of rank 0
 *   generating x and broadcasting it, EVERY rank independently calls
 *   genmat_random(x, n, 1, 43) and ends up holding a bit-identical copy of
 *   x, with zero messages sent. This only works for x because every rank
 *   wants the *entire* vector; it would NOT work for generating A's row
 *   blocks independently, because the PRNG is one sequential stream --
 *   asking rank 2 to "generate my slice of A" from scratch would just
 *   replay the *start* of the stream, not the correct offset into it. So A
 *   is still generated once, in full, on rank 0, and its row blocks are
 *   physically scattered to the other ranks below.
 *
 * WHY NO DERIVED MPI DATATYPE (unlike matmul's Cannon's algorithm):
 *
 *   A is stored row-major. Splitting by whole rows means each rank's slice
 *   is already one contiguous run of memory (row_start(r)*n .. up to
 *   local_rows(r)*n reals later) -- a plain MPI_Scatterv with per-rank
 *   element counts and byte offsets is enough. Cannon's needed
 *   MPI_Type_create_subarray because it split into 2D tiles, which are
 *   NOT contiguous in a row-major array. This kernel is simpler because
 *   the decomposition matches the storage order.
 *
 * TIMING: same hand-rolled convention as matmul/mpi.c, not bench_run() --
 *   that harness assumes one process on one core. MPI_Wtime() around just
 *   the local compute (not the one-time distribute/collect setup), then
 *   MPI_Allreduce(..., MPI_MAX, ...) per repetition, since the slowest rank
 *   gates real wall-clock time. Same discard-2-warm-ups, >=5 timed reps,
 *   report min/median discipline as bench.c, reimplemented by hand for the
 *   same reason as matmul/mpi.c.
 *
 * CHECKSUMS:
 *   - sum(y): every rank sums its own local piece, MPI_Reduce(..., SUM, ...)
 *     combines them at rank 0. No need to gather the full y vector just to
 *     add it up.
 *   - y[0]: row 0 is always on rank 0 under this distribution (rank 0's
 *     row_start is always 0), so no communication needed at all.
 *   - y[n-1]: computed on whichever rank owns the last row -- found by
 *     inverting the same block formula -- then sent to rank 0 with a single
 *     point-to-point MPI_Send/MPI_Recv, skipped entirely when that owner
 *     already IS rank 0 (e.g. p == 1, or p > n and rank 0 owns row 0 which
 *     also happens to be the only/last row).
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "genmat.h"
#include "bench.h"

#ifdef USE_DOUBLE
#define MPI_REAL_T MPI_DOUBLE
#else
#define MPI_REAL_T MPI_FLOAT
#endif

/* Returns how many rows rank `r` owns, out of `n` total rows over `p` ranks. */
static int rows_owned(int n, int p, int r)
{
    int base = n / p;
    int extra = n % p;
    return (r < extra) ? base + 1 : base;
}

/* Returns the index of the first row rank `r` owns. */
static int row_start_of(int n, int p, int r)
{
    int base = n / p;
    int extra = n % p;
    if (r < extra) {
        return r * (base + 1);
    }
    return extra * (base + 1) + (r - extra) * base;
}

/* Inverts the block formula: given a row index, which rank owns it? */
static int owner_of_row(int n, int p, int row)
{
    int base = n / p;
    int extra = n % p;
    int boundary = extra * (base + 1); /* rows before this are in the +1 group */
    if (row < boundary) {
        return row / (base + 1);
    }
    /* base == 0 can't reach here: boundary == extra*(1) == n covers every
     * valid row already when base == 0, so this branch is only taken when
     * base > 0. */
    return extra + (row - boundary) / base;
}

/* y_local = A_local * x, one dot product per locally-owned row. Identical
 * inner loop to serial.c's -- the parallelism here is entirely in *which
 * rows* a given rank owns, not in how a single row is computed. */
static void matvec_local(const real *A_local, const real *x, real *y_local,
                          int local_rows, int n)
{
    for (int i = 0; i < local_rows; i++) {
        const real *row = A_local + (size_t)i * n;
        real sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += row[j] * x[j];
        }
        y_local[i] = sum;
    }
}

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s -n <size> [-r <reps>]\n", prog);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    /* -t is meaningless here: rank count comes from `mpirun -np`, not a
     * CLI flag, same convention as matmul/mpi.c. */

    const int n = cfg.n;
    const int local_rows = rows_owned(n, size, rank);

    /* x: every rank generates its own identical copy, no communication. */
    real *x = malloc((size_t)n * sizeof(real));
    if (!x) { fprintf(stderr, "rank %d: alloc x failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    genmat_random(x, n, 1, 43);

    /* A: only rank 0 ever holds the full matrix. */
    real *A_full = NULL;
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        A_full = malloc((size_t)n * n * sizeof(real));
        if (!A_full) { fprintf(stderr, "rank 0: alloc A_full failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        genmat_random(A_full, n, n, 42);

        sendcounts = malloc((size_t)size * sizeof(int));
        displs = malloc((size_t)size * sizeof(int));
        for (int r = 0; r < size; r++) {
            sendcounts[r] = rows_owned(n, size, r) * n;   /* elements, not rows */
            displs[r]     = row_start_of(n, size, r) * n; /* elements, not rows */
        }
    }

    real *A_local = (local_rows > 0) ? malloc((size_t)local_rows * n * sizeof(real)) : NULL;
    real *y_local = (local_rows > 0) ? malloc((size_t)local_rows * sizeof(real)) : NULL;
    if (local_rows > 0 && (!A_local || !y_local)) {
        fprintf(stderr, "rank %d: alloc A_local/y_local failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* One-time distribution -- not part of the timed region, same as
     * matmul/mpi.c only timing the skew+shift loop, not the initial
     * MPI_Scatterv of A/B. */
    MPI_Scatterv(A_full, sendcounts, displs, MPI_REAL_T,
                 A_local, local_rows * n, MPI_REAL_T,
                 0, MPI_COMM_WORLD);

    /* 2 warm-up calls, discarded. */
    for (int w = 0; w < 2; w++) {
        matvec_local(A_local, x, y_local, local_rows, n);
    }

    const int reps = (cfg.reps > 0) ? cfg.reps : 5;
    double *rep_times = malloc((size_t)reps * sizeof(double));
    for (int i = 0; i < reps; i++) {
        MPI_Barrier(MPI_COMM_WORLD); /* start every rank's timer together */
        double t0 = MPI_Wtime();
        matvec_local(A_local, x, y_local, local_rows, n);
        double t1 = MPI_Wtime();
        double local_dt = t1 - t0;
        double global_dt;
        MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        rep_times[i] = global_dt;
    }

    /* min + median, identical result on every rank since Allreduce already
     * synced the per-rep values above -- only rank 0 will actually print
     * them, but there's no harm in every rank computing the same numbers. */
    double time_min = rep_times[0];
    for (int i = 1; i < reps; i++) if (rep_times[i] < time_min) time_min = rep_times[i];
    for (int i = 1; i < reps; i++) {
        double key = rep_times[i];
        int j = i - 1;
        while (j >= 0 && rep_times[j] > key) { rep_times[j + 1] = rep_times[j]; j--; }
        rep_times[j + 1] = key;
    }
    double time_med = rep_times[reps / 2];
    free(rep_times);

    /* sum(y): each rank reduces its own partial sum. */
    real local_sum = 0.0;
    for (int i = 0; i < local_rows; i++) local_sum += y_local[i];
    real global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_REAL_T, MPI_SUM, 0, MPI_COMM_WORLD);

    /* y[0]: always local to rank 0 under this distribution. */
    real y_first = 0.0;
    if (rank == 0) y_first = y_local[0];

    /* y[n-1]: fetched from whichever rank actually owns the last row. */
    int last_owner = owner_of_row(n, size, n - 1);
    real y_last = 0.0;
    if (rank == 0 && last_owner == 0) {
        y_last = y_local[local_rows - 1];
    } else if (rank == last_owner) {
        MPI_Send(&y_local[local_rows - 1], 1, MPI_REAL_T, 0, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        MPI_Recv(&y_last, 1, MPI_REAL_T, last_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        fprintf(stderr, "sum(y)=%.7e y[0]=%.7e y[n-1]=%.7e\n",
                (double)global_sum, (double)y_first, (double)y_last);

        double flops = 2.0 * (double)n * (double)n;
        double gflops = flops / time_min / 1e9;
        /* Same working-set-estimate caveat as serial.c's CSV -- even more
         * approximate here, since x is physically duplicated in every
         * rank's memory rather than read once. Deferred to Phase 5 like
         * every other kernel's gbytes_s. */
        double gbytes_s = ((double)n * n + 2.0 * n) * sizeof(real) / time_min / 1e9;

        bench_report_csv(stdout, "matvec", "mpi", n, size,
                          time_min, time_med, gflops, gbytes_s);
    }

    free(x);
    free(A_local);
    free(y_local);
    if (rank == 0) {
        free(A_full);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
