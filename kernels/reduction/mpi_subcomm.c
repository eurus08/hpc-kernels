/*
 * kernels/reduction/mpi_subcomm.c
 *
 * MPI reduction with a two-level hierarchical combine, instead of the
 * flat single MPI_Reduce that mpi.c used.
 *
 * Motivation: at real cluster scale, reducing directly across every
 * rank in MPI_COMM_WORLD means every rank talks to the root over
 * whatever network links separate them -- including slow cross-node
 * links. The standard fix is a two-level tree: split ranks into
 * groups (in production, one group per physical node, via
 * MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)), reduce within
 * each group first (fast, often shared-memory-adjacent), then have
 * only the group leaders reduce again across the (fewer, more
 * expensive) links between groups.
 *
 * On a single 6-core dev machine there is only one physical node, so
 * the grouping here is deliberately artificial (-g <num_groups>,
 * contiguous rank blocks) purely to exercise the real mechanics
 * (MPI_Comm_split, MPI_UNDEFINED, two-level MPI_Reduce) correctly.
 * The actual payoff -- node-local reduction being cheaper than
 * cross-node -- can't be demonstrated on one machine; that's a
 * Lengau-scale story, documented honestly rather than implied here.
 *
 * Data distribution (counts[]/displs[], Scatterv for sum mode, local
 * index ranges for pi mode) is identical to mpi.c -- only how the
 * partial sums get COMBINED changes.
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

typedef struct {
    const char *mode;
    int n;
    int rank;        /* world rank */
    const int *counts;
    const int *displs;
    real *full;
    real *local;
    MPI_Comm sub_comm;    /* this rank's group -- level 1 */
    MPI_Comm leader_comm; /* MPI_COMM_NULL unless this rank is a group leader -- level 2 */
    real result;          /* valid only on world rank 0, after both levels */
} reduce_arg;

static void reduce_kernel(void *arg_v) {
    reduce_arg *arg = (reduce_arg *)arg_v;
    int my_n = arg->counts[arg->rank];
    real local_sum = (real)0.0;

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

    /* Level 1: reduce within this rank's group. Only the group's own
     * rank 0 (the "group leader") ends up with a meaningful group_sum
     * -- every other rank's group_sum is unused/garbage, which is
     * fine, since only leaders proceed to level 2. */
    real group_sum = (real)0.0;
    MPI_Reduce(&local_sum, &group_sum, 1, MPI_REAL_T, MPI_SUM, 0, arg->sub_comm);

    /* Level 2: only group leaders participate -- everyone else got
     * MPI_COMM_NULL for leader_comm back when it was created, and
     * calling a collective on MPI_COMM_NULL is undefined behaviour,
     * so this must be guarded. */
    if (arg->leader_comm != MPI_COMM_NULL) {
        real global_sum = (real)0.0;
        MPI_Reduce(&group_sum, &global_sum, 1, MPI_REAL_T, MPI_SUM, 0, arg->leader_comm);
        if (arg->rank == 0) {
            arg->result = global_sum;
        }
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

/* Same removal trick as extract_mode, for an integer flag instead of a
 * string. -1 sentinel means "not given" so the caller can apply a
 * size-dependent default (we don't know world size until after
 * MPI_Comm_size, which runs before this is called). */
static int extract_groups(int *argc, char **argv) {
    int groups = -1;
    for (int i = 1; i < *argc; i++) {
        if (strcmp(argv[i], "-g") == 0 && i + 1 < *argc) {
            groups = atoi(argv[i + 1]);
            for (int j = i; j + 2 < *argc; j++) {
                argv[j] = argv[j + 2];
            }
            *argc -= 2;
            break;
        }
    }
    return groups;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *mode = extract_mode(&argc, argv);
    int num_groups = extract_groups(&argc, argv);
    if (num_groups < 0) {
        num_groups = (size >= 2) ? 2 : 1; /* default: 2 groups if possible */
    }
    if (num_groups < 1 || num_groups > size) {
        if (rank == 0) {
            fprintf(stderr, "error: -g must be between 1 and world size (%d), got %d\n",
                    size, num_groups);
        }
        MPI_Finalize();
        return 1;
    }

    bench_config cfg;
    if (bench_parse_args(argc, argv, &cfg) != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "usage: mpirun -np <ranks> %s -n <N> [-r <reps>] [-m sum|pi] [-g <num_groups>]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    /* Global block distribution -- identical to mpi.c, unaffected by
     * grouping. Grouping only changes how partial sums get combined. */
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

    /* Level-1 grouping: color = which contiguous block of ranks this
     * rank belongs to. Passing `rank` as the key keeps ordering within
     * each new group predictable (ascending world-rank order). In a
     * real multi-node deployment this color would come from
     * MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
     * MPI_INFO_NULL, &sub_comm) instead -- grouping by physical node
     * rather than an arbitrary block. */
    int color = (rank * num_groups) / size;
    MPI_Comm sub_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);

    int sub_rank;
    MPI_Comm_rank(sub_comm, &sub_rank);
    int is_leader = (sub_rank == 0);

    /* Level-2 grouping: only group leaders (sub_rank == 0) join a new
     * communicator together. Everyone else passes MPI_UNDEFINED and
     * gets MPI_COMM_NULL back -- they take no part in level 2. */
    MPI_Comm leader_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_leader ? 0 : MPI_UNDEFINED, rank, &leader_comm);

    reduce_arg arg;
    arg.mode = mode;
    arg.n = cfg.n;
    arg.rank = rank;
    arg.counts = counts;
    arg.displs = displs;
    arg.full = full;
    arg.local = local;
    arg.sub_comm = sub_comm;
    arg.leader_comm = leader_comm;
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

        bench_report_csv(stdout, "reduction", "mpi_subcomm", cfg.n, size,
                          global_min, global_med, gflops, gbytes_s);

        fprintf(stderr, "mode=%s ranks=%d groups=%d result=" REAL_FMT "\n",
                mode, size, num_groups, arg.result);

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

    MPI_Comm_free(&sub_comm);
    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
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
