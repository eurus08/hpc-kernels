/*
 * kernels/gaussian/mpi.c
 *
 * MPI Gaussian elimination with back-substitution, solving A x = b.
 *
 * DECOMPOSITION: CYCLIC, not block (confirmed with Prince before writing
 * this, and different from every other MPI kernel in this repo so far).
 *
 *   Row i is owned by rank (i mod p) -- so rank 0 owns rows 0, p, 2p, ...;
 *   rank 1 owns rows 1, p+1, 2p+1, ...; and so on. Contrast with matvec's
 *   block distribution, where rank 0 owned one contiguous chunk of early
 *   rows and the last rank owned a contiguous chunk of late rows.
 *
 *   WHY: elimination doesn't touch every row equally throughout the run.
 *   By pivot step k, rows 0..k are already finished (they were pivots or
 *   are otherwise untouched from here on); only rows k+1..n-1 still have
 *   real work left. Under BLOCK distribution, whichever rank owns the
 *   *early* rows runs out of useful work as k grows and sits idle for the
 *   rest of the algorithm -- exactly the load-imbalance failure mode that
 *   doesn't show up in matmul or matvec, where every row/tile carries
 *   equal work the whole time. Under CYCLIC distribution, every rank still
 *   owns roughly (n-k)/p of the remaining unprocessed rows at every step,
 *   no matter how far elimination has progressed -- the standard textbook
 *   answer to load-balancing elimination-shaped algorithms specifically.
 *
 * WHY NO MPI_Scatterv FOR THE INITIAL DISTRIBUTION (unlike matvec):
 *
 *   Matvec's row-blocks were physically contiguous in A's row-major
 *   layout, so a single MPI_Scatterv call handled it. Cyclic rows are NOT
 *   contiguous -- rank 0's rows (0, p, 2p, ...) are scattered throughout
 *   memory. Rather than reach for a strided MPI derived datatype
 *   (MPI_Type_vector) purely to speed up a one-time, untimed setup step,
 *   this file just loops over all n rows on rank 0 and sends each one,
 *   individually, to whichever rank owns it. Simpler to read, and it
 *   costs nothing in the numbers that actually matter, since this
 *   distribution step happens once and is never inside the timed region.
 *
 * PER-PIVOT COMMUNICATION (the actual interesting part):
 *
 *   Forward elimination, step k: the rank owning row k (rank k mod p)
 *   packs that row's coefficients PLUS its corresponding b[k] into one
 *   buffer, and MPI_Bcast's it to everyone (itself included, for code
 *   simplicity -- broadcasting to yourself is a no-op cost-wise). Every
 *   rank then eliminates using that buffer against whichever of ITS OWN
 *   locally-owned rows still lie below the pivot (global index > k).
 *
 *   Back-substitution runs the same broadcast pattern in reverse: solving
 *   for x[i] needs x[i+1..n-1], which under cyclic distribution are
 *   scattered across every rank, not sitting locally the way a block
 *   scheme would keep them. So going from i = n-1 down to 0, whichever
 *   rank owns row i computes x[i] and MPI_Bcast's just that one value to
 *   everyone. By the time the loop reaches i, every rank already has
 *   every x[i+1..n-1] value from earlier broadcasts. A side effect of
 *   this approach (confirmed with Prince as intentional, not accidental):
 *   by the end of the loop, EVERY rank holds the complete x vector, with
 *   no separate MPI_Gatherv needed at all.
 *
 * SAME "REPEATED CALLS ARE A FIXED POINT" PROPERTY AS serial.c:
 *
 *   No pivoting here either (same diagonal-dominance guarantee from
 *   common/genmat.h as serial.c), so once a rank's local rows are fully
 *   eliminated, calling elimination again computes factor = 0/pivot = 0
 *   for every remaining row and changes nothing -- safe to call this
 *   kernel repeatedly (warm-up + timed reps) without re-distributing A
 *   and b each time, exactly like serial.c.
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

/* How many of the n rows does rank r own, under i % p == r? */
static int local_row_count(int n, int p, int r)
{
    if (r >= n) return 0;
    return (n - r + p - 1) / p;
}

/* The global row index of rank r's local_idx'th owned row. */
static int global_row_of(int local_idx, int p, int r)
{
    return local_idx * p + r;
}

/* Inverse of the above, valid only when called by the rank that actually
 * owns `global_row` (i.e. global_row % p == that rank's own rank number). */
static int local_idx_of(int global_row, int p)
{
    return global_row / p;
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
    /* -t is meaningless here, same as every other MPI file -- rank count
     * comes from `mpirun -np`. */

    const int n = cfg.n;
    const int my_rows = local_row_count(n, size, rank);

    real *A_local = (my_rows > 0) ? malloc((size_t)my_rows * n * sizeof(real)) : NULL;
    real *b_local = (my_rows > 0) ? malloc((size_t)my_rows * sizeof(real)) : NULL;
    real *x_all   = malloc((size_t)n * sizeof(real)); /* every rank ends up with the full vector */
    real *xfer    = malloc((size_t)(n + 1) * sizeof(real)); /* one row + its b entry */
    if ((my_rows > 0 && (!A_local || !b_local)) || !x_all || !xfer) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* --- One-time distribution, not part of the timed region --- */
    real *A_full = NULL, *b_full = NULL;
    if (rank == 0) {
        A_full = malloc((size_t)n * n * sizeof(real));
        b_full = malloc((size_t)n * sizeof(real));
        if (!A_full || !b_full) { fprintf(stderr, "rank 0: alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        genmat_diag_dominant(A_full, n, 42);
        genmat_random(b_full, n, 1, 43);
    }

    for (int g = 0; g < n; g++) {
        const int owner = g % size;
        if (rank == 0) {
            for (int j = 0; j < n; j++) xfer[j] = A_full[(size_t)g * n + j];
            xfer[n] = b_full[g];
            if (owner == 0) {
                const int li = local_idx_of(g, size);
                for (int j = 0; j < n; j++) A_local[(size_t)li * n + j] = xfer[j];
                b_local[li] = xfer[n];
            } else {
                MPI_Send(xfer, n + 1, MPI_REAL_T, owner, g, MPI_COMM_WORLD);
            }
        } else if (rank == owner) {
            MPI_Recv(xfer, n + 1, MPI_REAL_T, 0, g, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const int li = local_idx_of(g, size);
            for (int j = 0; j < n; j++) A_local[(size_t)li * n + j] = xfer[j];
            b_local[li] = xfer[n];
        }
    }

    /* --- Timed region: forward elimination + back-substitution --- */
    const int reps = (cfg.reps > 0) ? cfg.reps : 5;
    double *rep_times = malloc((size_t)reps * sizeof(double));

    /* 2 warm-ups first, discarded -- safe to repeat thanks to the
     * fixed-point property explained at the top of the file. */
    for (int w = 0; w < 2 + reps; w++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /* Forward elimination */
        for (int k = 0; k < n - 1; k++) {
            const int owner = k % size;
            if (rank == owner) {
                const int li = local_idx_of(k, size);
                for (int j = 0; j < n; j++) xfer[j] = A_local[(size_t)li * n + j];
                xfer[n] = b_local[li];
            }
            MPI_Bcast(xfer, n + 1, MPI_REAL_T, owner, MPI_COMM_WORLD);

            const real pivot_val = xfer[k];
            for (int li = 0; li < my_rows; li++) {
                const int gi = global_row_of(li, size, rank);
                if (gi > k) {
                    const real factor = A_local[(size_t)li * n + k] / pivot_val;
                    for (int j = k; j < n; j++) {
                        A_local[(size_t)li * n + j] -= factor * xfer[j];
                    }
                    b_local[li] -= factor * xfer[n];
                }
            }
        }

        /* Back-substitution */
        for (int i = n - 1; i >= 0; i--) {
            const int owner = i % size;
            if (rank == owner) {
                const int li = local_idx_of(i, size);
                real sum = b_local[li];
                for (int j = i + 1; j < n; j++) {
                    sum -= A_local[(size_t)li * n + j] * x_all[j];
                }
                x_all[i] = sum / A_local[(size_t)li * n + i];
            }
            MPI_Bcast(&x_all[i], 1, MPI_REAL_T, owner, MPI_COMM_WORLD);
        }

        double t1 = MPI_Wtime();
        if (w >= 2) {
            double local_dt = t1 - t0;
            double global_dt;
            MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            rep_times[w - 2] = global_dt;
        }
    }

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

    if (rank == 0) {
        real sum_x = 0.0;
        for (int i = 0; i < n; i++) sum_x += x_all[i];
        fprintf(stderr, "sum(x)=%.7e x[0]=%.7e x[n-1]=%.7e\n",
                (double)sum_x, (double)x_all[0], (double)x_all[n - 1]);

        double flops = (2.0 / 3.0) * (double)n * (double)n * (double)n;
        double gflops = flops / time_min / 1e9;
        double gbytes_s = ((double)n * n + 2.0 * n) * sizeof(real) / time_min / 1e9;

        bench_report_csv(stdout, "gaussian", "mpi", n, size,
                          time_min, time_med, gflops, gbytes_s);
    }

    free(A_local);
    free(b_local);
    free(x_all);
    free(xfer);
    if (rank == 0) {
        free(A_full);
        free(b_full);
    }

    MPI_Finalize();
    return 0;
}
