CC      = gcc
MPICC   = mpicc
NVCC    = nvcc

CFLAGS  = -O3 -march=native -Icommon
OMPFLAGS= -fopenmp
NVFLAGS = -O3 -arch=sm_61 -Icommon

ifeq ($(PREC),double)
CFLAGS += -DUSE_DOUBLE
NVFLAGS += -DUSE_DOUBLE
endif

.PHONY: all clean reduction_serial reduction_omp reduction_mpi reduction_mpi_subcomm reduction_mpi_omp matmul_serial matmul_omp matmul_mpi matvec_serial matvec_mpi gaussian_serial gaussian_mpi transpose_serial transpose_omp reduce_atomic reduce_shared reduce_shuffle transpose_naive transpose_tiled jacobi saxpy

# Builds every wired binary in one go. Kept in sync by hand as new targets
# are added below — if you add a bin/... rule, add its short name here too.
all: reduction_serial reduction_omp reduction_mpi reduction_mpi_subcomm reduction_mpi_omp \
     matmul_serial matmul_omp matmul_mpi \
     matvec_serial matvec_mpi \
     gaussian_serial gaussian_mpi \
     transpose_serial transpose_omp \
     reduce_atomic reduce_shared reduce_shuffle \
     transpose_naive transpose_tiled \
     jacobi saxpy

bin/reduction_serial: kernels/reduction/serial.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o $@

reduction_serial: bin/reduction_serial

bin/reduction_omp: kernels/reduction/omp.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -lm -o $@

reduction_omp: bin/reduction_omp

bin/reduction_mpi: kernels/reduction/mpi.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -lm -o $@

reduction_mpi: bin/reduction_mpi

bin/reduction_mpi_subcomm: kernels/reduction/mpi_subcomm.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -lm -o $@

reduction_mpi_subcomm: bin/reduction_mpi_subcomm

bin/reduction_mpi_omp: kernels/reduction/mpi_omp.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $(OMPFLAGS) $^ -lm -o $@

reduction_mpi_omp: bin/reduction_mpi_omp

bin/matmul_serial: kernels/matmul/serial.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o $@

matmul_serial: bin/matmul_serial

bin/matmul_omp: kernels/matmul/omp.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -lm -o $@

matmul_omp: bin/matmul_omp

bin/matmul_mpi: kernels/matmul/mpi.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -lm -o $@

matmul_mpi: bin/matmul_mpi

bin/matvec_serial: kernels/matvec/serial.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o $@

matvec_serial: bin/matvec_serial

bin/matvec_mpi: kernels/matvec/mpi.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -lm -o $@

matvec_mpi: bin/matvec_mpi

bin/gaussian_serial: kernels/gaussian/serial.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o $@

gaussian_serial: bin/gaussian_serial

bin/gaussian_mpi: kernels/gaussian/mpi.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -lm -o $@

gaussian_mpi: bin/gaussian_mpi

bin/transpose_serial: kernels/transpose/serial.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o $@

transpose_serial: bin/transpose_serial

bin/transpose_omp: kernels/transpose/omp.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -lm -o $@

transpose_omp: bin/transpose_omp

# --- CUDA targets ---
# No common/verify.c here: the .cu files don't #include verify.h — their
# correctness is checked externally against reference/check_*.py oracles,
# not via a linked-in verify_result() call like the CPU kernels use.

bin/reduce_atomic: kernels/reduction/reduce_atomic.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

reduce_atomic: bin/reduce_atomic

bin/reduce_shared: kernels/reduction/reduce_shared.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

reduce_shared: bin/reduce_shared

bin/reduce_shuffle: kernels/reduction/reduce_shuffle.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

reduce_shuffle: bin/reduce_shuffle

bin/transpose_naive: kernels/transpose/transpose_naive.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

transpose_naive: bin/transpose_naive

bin/transpose_tiled: kernels/transpose/transpose_tiled.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

transpose_tiled: bin/transpose_tiled

bin/jacobi: kernels/jacobi/jacobi.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

jacobi: bin/jacobi

bin/saxpy: kernels/saxpy/saxpy.cu common/genmat.c common/bench.c
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $^ -lm -o $@

saxpy: bin/saxpy

clean:
	rm -f *.o
	rm -rf bin/
