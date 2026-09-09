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

.PHONY: all clean reduction_serial reduction_omp reduction_mpi reduction_mpi_subcomm reduction_mpi_omp matmul_serial matmul_omp matmul_mpi matvec_serial

all:
	@echo "Scaffolding only — no targets wired up yet (Phase 1)."

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

clean:
	rm -f *.o
	rm -rf bin/
