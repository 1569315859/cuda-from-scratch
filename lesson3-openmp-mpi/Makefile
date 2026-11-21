# 第 3 课 Makefile：make 编译全部；make bench 自动跑 1,2,4,...,nproc 线程/进程
CXX      = g++
MPICXX   = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
N       ?= 1024
NPROC   := $(shell nproc)
# 1 2 4 8 ... 直到不超过 nproc，再把 nproc 本身加上（去重）
COUNTS  := $(shell n=1; s=""; while [ $$n -le $(NPROC) ]; do s="$$s $$n"; n=$$((n*2)); done; \
             echo $$s $(NPROC) | tr ' ' '\n' | sort -n | uniq | tr '\n' ' ')

all: omp_sum omp_matmul mpi_matmul

omp_sum: omp_sum.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

omp_matmul: omp_matmul.cpp common.h
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

mpi_matmul: mpi_matmul.cpp common.h
	$(MPICXX) $(CXXFLAGS) $< -o $@

bench: all
	@echo "nproc = $(NPROC); counts = $(COUNTS)"
	@for t in $(COUNTS); do echo "=== OMP_NUM_THREADS=$$t ==="; \
	  OMP_NUM_THREADS=$$t ./omp_matmul $(N) 1; done
	@for p in $(COUNTS); do echo "=== mpirun -np $$p ==="; \
	  mpirun -np $$p --oversubscribe ./mpi_matmul $(N) 0; done

clean:
	rm -f omp_sum omp_matmul mpi_matmul

.PHONY: all bench clean
