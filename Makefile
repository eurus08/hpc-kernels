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

.PHONY: all clean reduction_serial

all:
	@echo "Scaffolding only — no targets wired up yet (Phase 1)."

bin/reduction_serial: kernels/reduction/serial.c common/genmat.c common/verify.c common/bench.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o $@

reduction_serial: bin/reduction_serial

clean:
	rm -f *.o
	rm -rf bin/
