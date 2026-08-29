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

.PHONY: all clean
all:
	@echo "Scaffolding only — no targets wired up yet (Phase 1)."

clean:
	rm -f *.o
	rm -rf bin/
