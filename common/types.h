#ifndef HPC_KERNELS_TYPES_H
#define HPC_KERNELS_TYPES_H

/*
 * Precision switch for the whole project.
 *
 * Every kernel uses `real`, never `double` or `float` directly.
 * That's what turns the FP32/FP64 comparison into a compile flag
 * (`make PREC=double`) instead of a rewrite.
 *
 * USE_DOUBLE is defined by the top-level Makefile when invoked as
 * `make PREC=double`. Left undefined (the default), everything
 * builds in single precision.
 */

#ifdef USE_DOUBLE
typedef double real;
#define REAL_FMT "%.15e"
#else
typedef float real;
#define REAL_FMT "%.7e"
#endif

#endif /* HPC_KERNELS_TYPES_H */
