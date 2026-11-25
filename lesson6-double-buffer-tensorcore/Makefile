# 第 6 课 Makefile
#   make              编译 sgemm6（需要 sm_80 及以上；你的 RTX 2050 是 sm_86）
#   make run          N=1024 和 N=2048 各跑一遍（GPU 唤醒 + 每版 1 次预热 + 10 次计时）
#   make sweep        N = 512 1024 2048 4096 扫一遍
#   make ptxas        每个 kernel 的寄存器 / 共享内存 / 溢出（第 3、9 章）
#   make sass         数机器指令：LDGSTS（cp.async）、LDSM（ldmatrix）、HMMA（Tensor Core）、LDS、FFMA ……（第 9 章）
#   make ncu-sol      SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / LaunchStats / WarpStateStats（第 9 章）
#   make ncu-metrics  直接取指标：共享内存、cp.async、Tensor 流水线利用率、warp 停顿原因（第 9 章）
#   make ncu-cublas   看 cuBLAS FP32 / FP16 用的是什么 kernel、Tensor 流水线忙不忙（第 9 章选做）
#   make ncu-query    在你的机器上查某个指标名存不存在（指标名对不上时用）
#   make racecheck    compute-sanitizer 查共享内存竞争：正常版 + 去掉 db_reg 同步的版本（第 10 章）
#   make cpwait-bug   去掉 cp.async wait 的版本：先直接跑看对错，再用 racecheck / synccheck 看（第 10 章）
#   make memcheck     compute-sanitizer 查越界（第 10 章）
#   make emu          在没有 GPU 的机器上用 CPU 线程模拟检查下标（第 10.4 节，需要 g++ 12+ / clang++ 15+）
#   make NO_CUBLAS=1  不链接 cuBLAS

NVCC      ?= nvcc
ARCH      ?= sm_86
NVFLAGS    = -O2 -arch=$(ARCH) -std=c++17 -lineinfo
N         ?= 1024
NO_CUBLAS ?= 0
CXX       ?= g++

ifeq ($(NO_CUBLAS),0)
  CUBLAS_DEF = -DUSE_CUBLAS
  CUBLAS_LIB = -lcublas
endif

# 只匹配我们自己写的 kernel（名字以 sgemm_ 开头），不包括 cuBLAS 内部的 kernel 和 convert_f32_to_f16
KREGEX     = regex:^sgemm_
ONLY_OURS  = --only=2d_vec,db_reg,db_cpasync,warptile,wmma_naive,wmma_fp16
ONLY_CP    = --only=db_cpasync,warptile,wmma_fp16

# 第 5 课用过的共享内存 / 指令指标
NCU_MET_SMEM = smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__warps_active.avg.pct_of_peak_sustained_active
# 第 6 课新增：cp.async（LDGSTS）条数、全局读请求、Tensor Core 指令条数和流水线利用率
NCU_MET_L6 = smsp__inst_executed_op_ldgsts.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts.sum,dram__bytes_read.sum,smsp__inst_executed_pipe_tensor_op_hmma.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
# warp 停顿原因（每发射一条指令，平均有几个 warp 停在这个原因上）
NCU_MET_STALL = smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio

all: sgemm6

sgemm6: sgemm6.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) $(CUBLAS_DEF) $< -o $@ $(CUBLAS_LIB)

# ---- 故意带 bug 的版本，只给 compute-sanitizer 演示用 ----
sgemm6_nowait: sgemm6.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DSKIP_CP_WAIT $< -o $@

sgemm6_nosync: sgemm6.cu common.h common.cuh
	$(NVCC) $(NVFLAGS) -DSKIP_DB_SYNC $< -o $@

run: sgemm6
	./sgemm6 1024
	./sgemm6 2048

sweep: sgemm6
	for n in 512 1024 2048 4096; do ./sgemm6 $$n; done

ptxas:
	$(NVCC) $(NVFLAGS) -Xptxas -v -c sgemm6.cu -o /dev/null

# 静态数每个 kernel 机器码里各类指令出现了几次。
# 跳过带 @!PT 的行：谓词 !PT 永远为假，这些是编译器插的、永远不执行的占位指令。
sass: sgemm6
	cuobjdump -sass sgemm6 | awk '/Function :/ {f=$$3} /@!PT/ {next} {for (i = 1; i <= NF; i++) if ($$i ~ /^(LDGSTS|LDGDEPBAR|DEPBAR|LDSM|LDS|STS|LDG|STG|FFMA|HMMA|BAR)/) {sub(/;$$/, "", $$i); c[f "  " $$i]++; break}} END {for (k in c) print k, c[k]}' | sort

# ---- Nsight Compute（WSL2 上要先在 Windows 端打开 GPU 性能计数器权限，见第 4 课 7.2）----
ncu-sol: sgemm6
	ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	    --section WarpStateStats -k $(KREGEX) ./sgemm6 $(N) --profile $(ONLY_OURS)

ncu-metrics: sgemm6
	ncu --metrics $(NCU_MET_SMEM),$(NCU_MET_L6),$(NCU_MET_STALL) -k $(KREGEX) ./sgemm6 $(N) --profile $(ONLY_OURS)

# cuBLAS 的 kernel 名字各版本不同，这里不过滤；convert_f32_to_f16 那两行忽略即可
ncu-cublas: sgemm6
	ncu --metrics smsp__inst_executed_pipe_tensor_op_hmma.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size,launch__block_size \
	    ./sgemm6 $(N) --profile --only=cublas,cublas_tf32,cublas_fp16

ncu-report: sgemm6
	ncu --set full -k $(KREGEX) -o sgemm6_report -f ./sgemm6 $(N) --profile $(ONLY_OURS)

ncu-query:
	ncu --query-metrics | grep -E "ldgsts|pipe_tensor_op_hmma|pipe_fma_cycles|issue_stalled_(long_scoreboard|barrier|short_scoreboard|mio_throttle)|wavefronts_mem_shared|bank_conflicts_pipe_lsu_mem_shared"

# ---- compute-sanitizer（第 10 章）----
racecheck: sgemm6 sgemm6_nosync
	compute-sanitizer --tool racecheck ./sgemm6 256 --profile --only=2d_vec,db_reg,db_cpasync,warptile,wmma_fp16
	-compute-sanitizer --tool racecheck ./sgemm6_nosync 256 --profile --only=db_reg

cpwait-bug: sgemm6_nowait
	-./sgemm6_nowait 1024 $(ONLY_CP)
	-compute-sanitizer --tool racecheck ./sgemm6_nowait 256 --profile $(ONLY_CP)
	-compute-sanitizer --tool synccheck ./sgemm6_nowait 256 --profile $(ONLY_CP)

memcheck: sgemm6
	compute-sanitizer --tool memcheck ./sgemm6 1024 --profile $(ONLY_OURS)

# ---- 没有 GPU 时的 CPU 模拟（第 10.4 节）----
emu: test_l6_emulation.cpp sgemm6.cu
	sed -n '/尺寸参数（和第 5 课一样）/,/GPU kernels 结束/p' sgemm6.cu > kernels6_extracted.inc
	$(CXX) -std=c++20 -O2 -pthread test_l6_emulation.cpp -o test_l6_emulation
	./test_l6_emulation
	$(CXX) -std=c++20 -O2 -pthread -DSKIP_CP_WAIT test_l6_emulation.cpp -o test_l6_emulation_nowait
	-./test_l6_emulation_nowait

clean:
	rm -f sgemm6 sgemm6_nowait sgemm6_nosync *.ncu-rep test_l6_emulation test_l6_emulation_nowait kernels6_extracted.inc

.PHONY: all run sweep ptxas sass ncu-sol ncu-metrics ncu-cublas ncu-report ncu-query racecheck cpwait-bug memcheck emu clean
