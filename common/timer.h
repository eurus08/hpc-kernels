#ifndef HPC_KERNELS_TIMER_H
#define HPC_KERNELS_TIMER_H

#include <time.h>

/*
 * Wall-clock timing via CLOCK_MONOTONIC.
 *
 * Deliberately NOT using clock() from <time.h>'s CPU-time family:
 * clock() measures CPU-busy time only, and silently under-reports
 * any time a thread spends idle/blocked/waiting — which is exactly
 * the time an OpenMP or MPI benchmark needs to capture honestly.
 *
 * CLOCK_MONOTONIC is wall-clock time that only ever moves forward,
 * immune to NTP/system-clock adjustments mid-run (unlike CLOCK_REALTIME).
 */

static inline double timer_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif /* HPC_KERNELS_TIMER_H */
