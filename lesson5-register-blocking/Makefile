# 第 5 课 Makefile
#   make              编译 sgemm5
#   make run          N=1024 和 N=2048 各跑一遍（GPU 唤醒 + 每版 1 次预热 + 10 次计时）
#   make sweep        N = 512 1024 2048 4096 扫一遍
#   make ptxas        看编译器给每个 kernel 分了多少寄存器、有没有溢出（spill）（第 2、7 章）
#   make sass         数每个 kernel 的机器指令：共享内存读 LDS / LDS.128、FFMA ……（第 1、8 章）
#   make ncu-sol      SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / LaunchStats 四节（第 8 章）
#   make ncu-metrics  直接取指标：共享内存指令数、bank 冲突、FFMA 数、寄存器、占用率、全局访存（第 8 章）
#   make ncu-warp     warp 停顿原因（WarpStateStats / SchedulerStats）（第 8 章选做）
#   make ncu-report   全套指标存成 sgemm5_report.ncu-rep
#   make ncu-query    在你的机器上查某个指标名存不存在（指标名对不上时用）
#   make racecheck    compute-sanitizer 查共享内存竞争（第 9 章）
#   make memcheck     compute-sanitizer 查越界（第 9、10 章）
#   make NO_CUBLAS=1  不链接 cuBLAS

NVCC      ?= nvcc
ARCH      ?= sm_86
NVFLAGS    = -O2 -arch=$(ARCH) -std=c++17 -lineinfo
N         ?= 1024
NO_CUBLAS ?= 0

ifeq ($(NO_CUBLAS),0)
  CUBLAS_DEF = -DUSE_CUBLAS
  CUBLAS_LIB = -lcublas
endif

# 只匹配我们自己写的 kernel（名字以 sgemm_ 开头），不包括 cuBLAS 内部的 kernel
KREGEX     = regex:^sgemm_
ONLY_OURS  = --only=tile16,1d_tm8,2d_8x8,2d_vec

# 第 4 课用过的指标：全局读请求 / sector、命中率、DRAM 字节、吞吐、实际占用率
NCU_MET_L4 = l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
# 第 5 课新增：共享内存读写指令数、wavefront 数、bank 冲突、FFMA 条数、全部指令数、寄存器、寄存器限制的 block 数
NCU_MET_L5 = smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed.sum,launch__registers_per_thread,launch__occupancy_limit_registers

all: sgemm5

sgemm5: sgemm5.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) $(CUBLAS_DEF) $< -o $@ $(CUBLAS_LIB)

# ---- 故意带 bug 的版本，只给 compute-sanitizer 演示用 ----
sgemm5_nosync: sgemm5.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DSKIP_SECOND_SYNC $< -o $@

sgemm5_nosync1: sgemm5.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DSKIP_FIRST_SYNC $< -o $@

sgemm5_nocheck: sgemm5.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DNO_N_CHECK $< -o $@

run: sgemm5
	./sgemm5 1024
	./sgemm5 2048

sweep: sgemm5
	for n in 512 1024 2048 4096; do ./sgemm5 $$n; done

# 每个 kernel 打印一行 "Used NN registers, ... bytes smem"，以及 "bytes spill stores / spill loads"
ptxas:
	$(NVCC) $(NVFLAGS) -Xptxas -v -c sgemm5.cu -o /dev/null

# 静态数一数每个 kernel 机器码里各类指令出现了几次（循环体已被展开，所以是「一轮」的量）
sass: sgemm5
	cuobjdump -sass sgemm5 | awk '/Function :/ {f=$$3} {for (i = 1; i <= NF; i++) if ($$i ~ /^(LDS|STS|LDG|STG|FFMA|BAR)/) {sub(/;$$/, "", $$i); c[f "  " $$i]++; break}} END {for (k in c) print k, c[k]}' | sort

# ---- Nsight Compute（WSL2 上要先在 Windows 端打开 GPU 性能计数器权限，见第 4 课 7.2）----
ncu-sol: sgemm5
	ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	    -k $(KREGEX) ./sgemm5 $(N) --profile $(ONLY_OURS)

ncu-metrics: sgemm5
	ncu --metrics $(NCU_MET_L5),$(NCU_MET_L4) -k $(KREGEX) ./sgemm5 $(N) --profile $(ONLY_OURS)

ncu-warp: sgemm5
	ncu --section WarpStateStats --section SchedulerStats -k $(KREGEX) ./sgemm5 $(N) --profile $(ONLY_OURS)

ncu-report: sgemm5
	ncu --set full -k $(KREGEX) -o sgemm5_report -f ./sgemm5 $(N) --profile $(ONLY_OURS)

ncu-query:
	ncu --query-metrics | grep -E "inst_executed_op_shared|bank_conflicts_pipe_lsu_mem_shared|wavefronts_mem_shared|ffma_pred_on|registers_per_thread|occupancy_limit_registers"

# ---- compute-sanitizer（第 9、10 章）----
racecheck: sgemm5 sgemm5_nosync
	compute-sanitizer --tool racecheck ./sgemm5 256 --profile $(ONLY_OURS)
	-compute-sanitizer --tool racecheck ./sgemm5_nosync 256 --profile $(ONLY_OURS)

memcheck: sgemm5 sgemm5_nocheck
	compute-sanitizer --tool memcheck ./sgemm5 1024 --profile $(ONLY_OURS)
	-compute-sanitizer --tool memcheck ./sgemm5_nocheck 1000 --profile --only=1d_tm8,2d_8x8,2d_vec

clean:
	rm -f sgemm5 sgemm5_nosync sgemm5_nosync1 sgemm5_nocheck *.ncu-rep

.PHONY: all run sweep ptxas sass ncu-sol ncu-metrics ncu-warp ncu-report ncu-query racecheck memcheck clean
