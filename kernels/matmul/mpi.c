/*
 * kernels/matmul/mpi.c
 *
 * MPI dense matrix-matrix multiply: C = A * B, for square N x N
 * matrices, using Cannon's algorithm on a square 2D grid of ranks.
 *
 * Constraints (validated at startup, rejected with a clear error
 * otherwise -- no fallback path, deliberately):
 *   - World size p MUST be a perfect square: p = q*q. Ranks are
 *     arranged in a q x q grid.
 *   - N MUST be divisible by q, so every rank's local block is
 *     exactly the same size (nb = N/q) -- no padding/uneven-block
 *     logic in this file.
 *
 * ----------------------------------------------------------------
 * BEGINNER PRIMER: what Cannon's algorithm is doing and why
 * ----------------------------------------------------------------
 * serial.c and omp.c both run on ONE machine with shared memory --
 * any thread can read any part of A or B directly. MPI ranks are
 * separate processes (possibly on separate machines) with their own
 * PRIVATE memory -- there is no shared array to read from. So A and
 * B first have to be split into blocks and handed out to ranks, and
 * since computing a block of C needs matching blocks of BOTH A and
 * B, and no single rank holds all of either, ranks have to exchange
 * blocks with each other during the computation.
 *
 * Cannon's algorithm does that exchange with minimal data movement:
 *
 *   1. Arrange ranks into a q x q grid. Rank (i,j) starts holding
 *      block A[i][j] and block B[i][j] (a "block" here is an
 *      nb x nb chunk of the full N x N matrix, nb = N/q).
 *   2. INITIAL SKEW: shift row i of the A blocks left by i positions
 *      (wrapping around), and shift column j of the B blocks up by j
 *      positions (wrapping around). This lines things up so that,
 *      right after skewing, rank (i,j) already holds a matching pair
 *      of A/B blocks it can multiply together immediately.
 *   3. Repeat q times: multiply-accumulate your current local A
 *      block times local B block into your local C block, then shift
 *      your A block one step left (wrapping) and your B block one
 *      step up (wrapping) to your neighbors.
 *
 * After q rounds, every rank's local C block is finished -- no rank
 * ever needed the WHOLE matrix, just a rotating set of neighbors'
 * blocks. The wraparound shifts are why this needs a periodic 2D
 * Cartesian communicator (MPI_Cart_create with periods={1,1}) --
 * that's the MPI feature that turns "shift left, wrapping from
 * column 0 back to the last column" into a single library call.
 * ----------------------------------------------------------------
 */

#include "types.h"
#include "genmat.h"
#include "bench.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_DOUBLE
#define MPI_REAL_T MPI_DOUBLE
#else
#define MPI_REAL_T MPI_FLOAT
#endif

/* Local nb x nb block multiply-ACCUMULATE (does not zero C first --
 * the caller zeroes once per repetition, before the q-step loop,
 * since C accumulates contributions across all q rounds). Same ikj
 * ordering as serial.c/omp.c, just operating on a block instead of
 * the whole matrix. */
static void block_matmul_accumulate(const real *A, const real *B, real *C, int nb) {
    for (int i = 0; i < nb; i++) {
        for (int k = 0; k < nb; k++) {
            real a_ik = A[i * nb + k];
            for (int j = 0; j < nb; j++) {
                C[i * nb + j] += a_ik * B[k * nb + j];
            }
        }
    }
}

/* qsort comparator for computing the median of the timed reps. */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <p> %s -n <N> [-r <reps>]\n", argv[0]);
            fprintf(stderr, "  -n N is required: multiplies two N x N matrices.\n");
            fprintf(stderr, "  p (from -np) MUST be a perfect square (4, 9, 16, ...).\n");
            fprintf(stderr, "  N MUST be divisible by sqrt(p).\n");
            fprintf(stderr, "  -t is not used here -- parallelism comes from -np, not -t.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = cfg.n;

    /* q = sqrt(world_size), validated to be an EXACT integer square
     * root -- Cannon's algorithm needs a square grid, no fallback. */
    int q = (int)(sqrt((double)world_size) + 0.5);
    if (q * q != world_size) {
        if (world_rank == 0) {
            fprintf(stderr,
                "error: world size %d is not a perfect square. "
                "Cannon's algorithm needs mpirun -np <q*q> (e.g. 4, 9, 16).\n",
                world_size);
        }
        MPI_Finalize();
        return 1;
    }

    if (n % q != 0) {
        if (world_rank == 0) {
            fprintf(stderr,
                "error: N=%d is not divisible by q=%d (grid dimension, sqrt(%d) ranks). "
                "Every rank's block must be the same size -- choose an N divisible by %d.\n",
                n, q, world_size, q);
        }
        MPI_Finalize();
        return 1;
    }

    int nb = n / q; /* block dimension: each rank holds an nb x nb piece of A, B, C */

    /* 2D Cartesian communicator, periodic in both dimensions (needed
     * for the wraparound shifts). reorder=0 keeps each rank's cart
     * rank identical to its original MPI_COMM_WORLD rank -- so "rank
     * 0" means the same thing everywhere in this file, and rank 0
     * always sits at grid coordinates (0,0). */
    int dims[2] = { q, q };
    int periods[2] = { 1, 1 };
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int my_i = coords[0], my_j = coords[1];

    /* --- Build the "block" MPI datatype used to Scatterv the full
     * N x N matrices out to each rank's nb x nb piece. ---
     *
     * A block sitting inside a full N x N row-major array is NOT
     * contiguous in memory: it's nb separate short rows, each nb
     * elements long, with (N - nb) elements of SOMEONE ELSE'S row
     * data in between. MPI_Type_create_subarray describes that
     * pattern once ("an nb x nb window inside an N x N array"), so a
     * single MPI_Scatterv call can hand out every rank's block
     * without any manual row-by-row packing loop.
     *
     * The MPI_Type_create_resized call afterward is a standard, if
     * slightly arcane, companion step: by default the "extent" of the
     * subarray type (the stride MPI_Scatterv uses to figure out where
     * the NEXT block starts) would be based on the full nb-rows-times-
     * N-wide span the subarray touches -- not what we want, since
     * blocks are laid out at arbitrary (row,col) offsets, not one
     * after another. Resizing the type's extent down to sizeof(real)
     * (one element) lets us instead specify each block's position
     * directly, as a plain element-count displacement (row*N + col)
     * into the full array -- which is exactly the offset arithmetic
     * we already know how to do.
     */
    MPI_Datatype blocktype_tmp, blocktype;
    int sizes[2] = { n, n };
    int subsizes[2] = { nb, nb };
    int starts[2] = { 0, 0 };
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_REAL_T, &blocktype_tmp);
    MPI_Type_create_resized(blocktype_tmp, 0, (MPI_Aint)sizeof(real), &blocktype);
    MPI_Type_commit(&blocktype);

    /* Rank 0 generates the full matrices -- same seeds (42, 43) as
     * every other paradigm and the oracle, so all of them multiply
     * identical inputs. Every other rank leaves these NULL; they're
     * only read by MPI_Scatterv on the root. */
    real *A_full = NULL, *B_full = NULL;
    if (cart_rank == 0) {
        A_full = (real *)malloc((size_t)n * (size_t)n * sizeof(real));
        B_full = (real *)malloc((size_t)n * (size_t)n * sizeof(real));
        genmat_random(A_full, n, n, 42u);
        genmat_random(B_full, n, n, 43u);
    }

    /* sendcounts/displs computed identically (and independently, no
     * communication needed) on every rank -- MPI_Cart_coords is a
     * purely local lookup into the communicator's cached topology, so
     * every rank can work out where every OTHER rank's block sits.
     * Only rank 0's copies actually get used by Scatterv, but
     * computing them everywhere avoids a branch. */
    int *sendcounts = (int *)malloc((size_t)world_size * sizeof(int));
    int *displs = (int *)malloc((size_t)world_size * sizeof(int));
    for (int r = 0; r < world_size; r++) {
        int rc[2];
        MPI_Cart_coords(cart_comm, r, 2, rc);
        sendcounts[r] = 1; /* one "blocktype" instance per rank */
        displs[r] = rc[0] * nb * n + rc[1] * nb; /* element offset, in units of sizeof(real) */
    }

    real *A_orig = (real *)malloc((size_t)nb * (size_t)nb * sizeof(real));
    real *B_orig = (real *)malloc((size_t)nb * (size_t)nb * sizeof(real));
    real *A_work = (real *)malloc((size_t)nb * (size_t)nb * sizeof(real));
    real *B_work = (real *)malloc((size_t)nb * (size_t)nb * sizeof(real));
    real *C_local = (real *)malloc((size_t)nb * (size_t)nb * sizeof(real));

    MPI_Scatterv(A_full, sendcounts, displs, blocktype, A_orig, nb * nb, MPI_REAL_T, 0, cart_comm);
    MPI_Scatterv(B_full, sendcounts, displs, blocktype, B_orig, nb * nb, MPI_REAL_T, 0, cart_comm);

    if (cart_rank == 0) {
        free(A_full);
        free(B_full);
    }

    /* --- Timed repetitions ---
     * Not using common/bench.c's bench_run here: that harness assumes
     * a single-process wall clock, which doesn't make sense once
     * multiple ranks (possibly on different machines) are involved.
     * Instead: MPI_Wtime around the skew+shift-loop on every rank,
     * combined via MPI_Allreduce(..., MPI_MAX, ...) -- since ranks
     * proceed in lockstep (every shift is a blocking call), the
     * slowest rank each round is what actually gates wall-clock time,
     * which is the standard convention for reporting MPI benchmark
     * times. bench_report_csv() is still used afterward, purely for
     * uniform CSV formatting across every paradigm's output.
     *
     * Same "discard 2 warm-ups, keep >=5 timed reps, report min and
     * median" convention as common/bench.c, reimplemented here by
     * hand for the same reason. */
    const int WARMUP_REPS = 2;
    double *rep_times = (double *)malloc((size_t)cfg.reps * sizeof(double));

    for (int rep = -WARMUP_REPS; rep < cfg.reps; rep++) {
        memcpy(A_work, A_orig, (size_t)nb * nb * sizeof(real));
        memcpy(B_work, B_orig, (size_t)nb * nb * sizeof(real));
        memset(C_local, 0, (size_t)nb * nb * sizeof(real));

        MPI_Barrier(cart_comm); /* line everyone up before starting the clock */
        double t0 = MPI_Wtime();

        /* --- Initial skew ---
         * Row i's A block needs to end up shifted left by i; column
         * j's B block needs to end up shifted up by j. Unlike the
         * per-step rotation below, the shift amount here varies by
         * row/column, so we compute exact source/destination
         * coordinates directly (via MPI_Cart_rank) rather than using
         * a single uniform MPI_Cart_shift call. */
        {
            int dest_coords[2] = { my_i, ((my_j - my_i) % q + q) % q };
            int src_coords[2]  = { my_i, ((my_j + my_i) % q + q) % q };
            int dest_rank, src_rank;
            MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
            MPI_Cart_rank(cart_comm, src_coords, &src_rank);
            MPI_Sendrecv_replace(A_work, nb * nb, MPI_REAL_T,
                                  dest_rank, 100, src_rank, 100,
                                  cart_comm, MPI_STATUS_IGNORE);
        }
        {
            int dest_coords[2] = { ((my_i - my_j) % q + q) % q, my_j };
            int src_coords[2]  = { ((my_i + my_j) % q + q) % q, my_j };
            int dest_rank, src_rank;
            MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
            MPI_Cart_rank(cart_comm, src_coords, &src_rank);
            MPI_Sendrecv_replace(B_work, nb * nb, MPI_REAL_T,
                                  dest_rank, 200, src_rank, 200,
                                  cart_comm, MPI_STATUS_IGNORE);
        }

        /* --- q rounds of multiply, then shift (except after the
         * last round) --- */
        for (int step = 0; step < q; step++) {
            block_matmul_accumulate(A_work, B_work, C_local, nb);

            if (step < q - 1) {
                /* Uniform +/-1 shift this time -- MPI_Cart_shift is the
                 * right tool here, unlike the initial skew above.
                 * disp=-1 along the "column" axis (1) means: dest =
                 * rank at (i, j-1), source = rank at (i, j+1) -- i.e.
                 * I send my current A block to my LEFT neighbor and
                 * receive a new one from my RIGHT neighbor, which is
                 * exactly "every A block moves one step left,
                 * wrapping." Same logic for B, shifting along the
                 * "row" axis (0) instead, for "every B block moves one
                 * step up." */
                int a_src, a_dest, b_src, b_dest;
                MPI_Cart_shift(cart_comm, 1, -1, &a_src, &a_dest);
                MPI_Cart_shift(cart_comm, 0, -1, &b_src, &b_dest);
                MPI_Sendrecv_replace(A_work, nb * nb, MPI_REAL_T,
                                      a_dest, 300, a_src, 300,
                                      cart_comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv_replace(B_work, nb * nb, MPI_REAL_T,
                                      b_dest, 400, b_src, 400,
                                      cart_comm, MPI_STATUS_IGNORE);
            }
        }

        double t1 = MPI_Wtime();
        double local_elapsed = t1 - t0;
        double rep_time;
        MPI_Allreduce(&local_elapsed, &rep_time, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        if (rep >= 0) {
            rep_times[rep] = rep_time;
        }
    }

    qsort(rep_times, (size_t)cfg.reps, sizeof(double), cmp_double);
    double time_min = rep_times[0];
    double time_med = rep_times[cfg.reps / 2];
    if (cfg.reps % 2 == 0) {
        time_med = (rep_times[cfg.reps / 2 - 1] + rep_times[cfg.reps / 2]) / 2.0;
    }

    /* --- Checksum, gathered to rank 0 for the sanity line ---
     * sum(C): every rank sums its own nb*nb block, MPI_Reduce combines
     * them with MPI_SUM. C[0][0]: always lives on rank (0,0) = cart
     * rank 0 -- no communication needed, it's already there. C[N-1]
     * [N-1]: lives on whichever rank owns grid block (q-1,q-1); that
     * specific rank sends the single value to rank 0 with a plain
     * point-to-point message (skipped entirely if q==1, where rank 0
     * owns everything already). */
    real local_sum = (real)0.0;
    for (int idx = 0; idx < nb * nb; idx++) {
        local_sum += C_local[idx];
    }
    real global_sum = (real)0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_REAL_T, MPI_SUM, 0, cart_comm);

    int last_coords[2] = { q - 1, q - 1 };
    int last_rank;
    MPI_Cart_rank(cart_comm, last_coords, &last_rank);
    real corner_val = (real)0.0;
    if (last_rank == 0) {
        if (cart_rank == 0) {
            corner_val = C_local[(nb - 1) * nb + (nb - 1)];
        }
    } else {
        if (cart_rank == last_rank) {
            real v = C_local[(nb - 1) * nb + (nb - 1)];
            MPI_Send(&v, 1, MPI_REAL_T, 0, 500, cart_comm);
        } else if (cart_rank == 0) {
            MPI_Recv(&corner_val, 1, MPI_REAL_T, last_rank, 500, cart_comm, MPI_STATUS_IGNORE);
        }
    }

    if (cart_rank == 0) {
        double flops = 2.0 * (double)n * (double)n * (double)n;
        double gflops = flops / time_med / 1e9;

        /* Same working-set-estimate caveat as serial.c/omp.c -- see
         * serial.c's comment for the full explanation. Not measured
         * traffic; also doesn't count the inter-rank shift traffic
         * this file introduces, which serial/omp don't have at all --
         * a further reason this figure is a rough floor, not ground
         * truth, worth revisiting properly at Phase 5. */
        double bytes = 3.0 * (double)n * (double)n * (double)sizeof(real);
        double gbytes_s = bytes / time_med / 1e9;

        bench_report_csv(stdout, "matmul", "mpi", n, world_size,
                          time_min, time_med, gflops, gbytes_s);

        fprintf(stderr, "sanity: sum(C) = " REAL_FMT ", C[0][0] = " REAL_FMT ", C[%d][%d] = " REAL_FMT "\n",
                global_sum, C_local[0], n - 1, n - 1, corner_val);
    }

    MPI_Type_free(&blocktype);
    MPI_Type_free(&blocktype_tmp);
    free(sendcounts);
    free(displs);
    free(A_orig);
    free(B_orig);
    free(A_work);
    free(B_work);
    free(C_local);
    free(rep_times);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}
