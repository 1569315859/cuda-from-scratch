# 第 4 课 Makefile
#   make              编译 sgemm（GPU）和 cache_order（CPU）
#   make run          N=1024 和 N=1000 各跑一遍（1 次预热 + 10 次计时）
#   make ncu-basic    Nsight Compute 基础报告（第 6 章）
#   make ncu-sol      只看 SpeedOfLight / MemoryWorkloadAnalysis / Occupancy 三节
#   make ncu-metrics  直接取几个关键指标（合并访问、命中率、DRAM 字节数、占用率）
#   make memcheck     compute-sanitizer 查越界（第 8 章）
#   make racecheck    compute-sanitizer 查共享内存竞争（第 8 章）
#   make cachegrind   valgrind 模拟 CPU 缓存，对比 ijk / ikj（第 7 章）
#   make perf         perf stat 读 CPU 硬件计数器（WSL2 上很可能不支持，见第 7 章）
#   make NO_CUBLAS=1  不链接 cuBLAS（cuBLAS 装不上时用）

NVCC      ?= nvcc
ARCH      ?= sm_86
# -lineinfo：把「机器指令 ↔ 源代码行号」的对应关系写进程序，ncu / compute-sanitizer 才能指出是哪一行
NVFLAGS    = -O2 -arch=$(ARCH) -std=c++17 -lineinfo
CXX        = g++
CXXFLAGS   = -O2 -std=c++17 -Wall -Wextra -g
N         ?= 1024
NO_CUBLAS ?= 0

ifeq ($(NO_CUBLAS),0)
  CUBLAS_DEF = -DUSE_CUBLAS
  CUBLAS_LIB = -lcublas
endif

# 只匹配我们自己写的 kernel（名字以 sgemm_ 开头），不包括 cuBLAS 内部的 kernel
KREGEX     = regex:^sgemm_
NCU_MET    = l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active

all: sgemm cache_order

sgemm: sgemm.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) $(CUBLAS_DEF) $< -o $@ $(CUBLAS_LIB)

# ---- 故意带 bug 的版本，只给 compute-sanitizer 演示用 ----
sgemm_nosync: sgemm.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DSKIP_SECOND_SYNC $< -o $@

sgemm_nosync1: sgemm.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DSKIP_FIRST_SYNC $< -o $@

sgemm_nobounds: sgemm.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DNO_BOUNDS $< -o $@

cache_order: cache_order.cpp common.h
	$(CXX) $(CXXFLAGS) $< -o $@

run: sgemm
	./sgemm $(N)
	./sgemm 1000

# ---- Nsight Compute（需要先在 Windows 端打开 GPU 性能计数器权限，见讲义 6.2）----
ncu-basic: sgemm
	ncu --set basic -k $(KREGEX) ./sgemm $(N) --profile

ncu-sol: sgemm
	ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
	    -k $(KREGEX) ./sgemm $(N) --profile

ncu-metrics: sgemm
	ncu --metrics $(NCU_MET) -k $(KREGEX) ./sgemm $(N) --profile

# 保存成报告文件，之后可以用图形界面 ncu-ui 打开，或 ncu -i 在终端里看
ncu-report: sgemm
	ncu --set full -k $(KREGEX) -o sgemm_report -f ./sgemm $(N) --profile

# ---- compute-sanitizer（第 8 章）----
memcheck: sgemm sgemm_nobounds
	compute-sanitizer --tool memcheck ./sgemm 1000 --profile --only=naive,tile16,tile32
	-compute-sanitizer --tool memcheck ./sgemm_nobounds 1000 --profile --only=tile16,tile32

racecheck: sgemm sgemm_nosync
	compute-sanitizer --tool racecheck ./sgemm 256 --profile --only=tile16,tile32
	-compute-sanitizer --tool racecheck ./sgemm_nosync 256 --profile --only=tile16,tile32

# ---- CPU 缓存工具（第 7 章）----
# 注意：Valgrind 3.21 起 cachegrind 默认不模拟缓存，必须显式写 --cache-sim=yes
cachegrind: cache_order
	valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=cg.ijk ./cache_order ijk 256
	valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=cg.ikj ./cache_order ikj 256

perf: cache_order
	-perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./cache_order ijk 1024
	-perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./cache_order ikj 1024

clean:
	rm -f sgemm sgemm_nosync sgemm_nosync1 sgemm_nobounds cache_order cg.* *.ncu-rep

.PHONY: all run ncu-basic ncu-sol ncu-metrics ncu-report memcheck racecheck cachegrind perf clean
