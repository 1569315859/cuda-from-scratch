# 第 7 课 Makefile —— driver API、context、module、启动路径
#   make              编译全部程序 + 三种 GPU 代码文件（vadd.ptx / vadd.cubin / vadd.fatbin）
#   make run1         实验 1：纯 driver API，分别加载 cubin / ptx / fatbin / library（第 3、4 章）
#   make jit          实验 1b：只带 PTX 的 fatbin，有 / 没有 JIT 缓存各跑一次（第 3.4 节）
#   make inspect      用 cuobjdump 看三种文件里装了什么（第 3 章）
#   make ctx          实验 2：context（第 2 章）
#   make overhead     实验 3：启动开销，LAZY / EAGER 各一次（第 5 章）
#   make run-errors   实验 4：粘性 / 非粘性错误（第 6 章）
#   make procaddr     实验 5：dlsym vs cuGetProcAddress（第 7 章）
#   make hook         实验 6：LD_PRELOAD 拦截 cuLaunchKernel（第 7 章，第 10 课预告）
#   make strace       数一数启动路径上的系统调用（第 5.3 节；需要 strace）
#   make nsys         用 Nsight Systems 看 API 时间线（第 5 章选做）

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++
ARCH      ?= sm_86
NVFLAGS    = -O2 -arch=$(ARCH) -std=c++17
# -lcuda：driver API 的库。WSL2 上真正的 libcuda.so.1 在 /usr/lib/wsl/lib；链接时用工具包里的 stub 也行
LDCUDA     = -L$(CUDA_HOME)/lib64/stubs -L/usr/lib/wsl/lib -lcuda

PROGS   = drv_vadd ctx_explore launch_overhead errors proc_addr libhook_preview.so
MODULES = vadd.ptx vadd.cubin vadd.fatbin vadd_ptxonly.fatbin

all: $(PROGS) $(MODULES)

# ---- GPU 代码文件 ----
vadd.ptx: vadd_kernel.cu
	$(NVCC) -arch=$(ARCH) -ptx $< -o $@
vadd.cubin: vadd_kernel.cu
	$(NVCC) -arch=$(ARCH) -cubin $< -o $@
# fatbin 里放两份：sm_86 的 SASS + compute_86 的 PTX（备用）
vadd.fatbin: vadd_kernel.cu
	$(NVCC) -fatbin -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 $< -o $@
# 只有 PTX（compute_75），没有任何 SASS —— 驱动必须现场 JIT
vadd_ptxonly.fatbin: vadd_kernel.cu
	$(NVCC) -fatbin -gencode arch=compute_75,code=compute_75 $< -o $@

# ---- 程序 ----
drv_vadd: drv_vadd.cpp common7.h
	$(CXX) -O2 -std=c++17 -I$(CUDA_HOME)/include $< -o $@ $(LDCUDA)
ctx_explore: ctx_explore.cu common7.h
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDCUDA)
launch_overhead: launch_overhead.cu common7.h
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDCUDA)
errors: errors.cu common7.h
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDCUDA)
proc_addr: proc_addr.cu common7.h
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDCUDA) -ldl
libhook_preview.so: hook_preview.cpp
	$(CXX) -O2 -shared -fPIC -I$(CUDA_HOME)/include $< -o $@ -ldl

# ---- 实验 ----
run1: drv_vadd $(MODULES)
	@echo "===== cubin ====="; ./drv_vadd vadd.cubin
	@echo "===== ptx（JIT）====="; ./drv_vadd vadd.ptx
	@echo "===== fatbin ====="; ./drv_vadd vadd.fatbin
	@echo "===== cubin + cuLibraryLoadFromFile ====="; ./drv_vadd vadd.cubin library

# JIT 缓存默认在 ~/.nv/ComputeCache。第一次：关掉缓存，强制现场编译；第二次：开着缓存跑两遍
jit: drv_vadd vadd_ptxonly.fatbin
	@echo "===== 只有 PTX，禁用 JIT 缓存 ====="; CUDA_CACHE_DISABLE=1 ./drv_vadd vadd_ptxonly.fatbin | grep -E "加载|SASS"
	@echo "===== 只有 PTX，用缓存（第 1 遍可能写缓存）====="; ./drv_vadd vadd_ptxonly.fatbin | grep -E "加载|SASS"
	@echo "===== 只有 PTX，用缓存（第 2 遍应命中）====="; ./drv_vadd vadd_ptxonly.fatbin | grep -E "加载|SASS"
	@echo "===== 有 sm_86 SASS 的 cubin，对照 ====="; ./drv_vadd vadd.cubin | grep -E "加载|SASS"

inspect: $(MODULES) launch_overhead
	@echo "===== vadd.fatbin 里的 ELF(SASS) 和 PTX ====="; $(CUDA_HOME)/bin/cuobjdump --list-elf --list-ptx vadd.fatbin
	@echo "===== vadd_ptxonly.fatbin ====="; $(CUDA_HOME)/bin/cuobjdump --list-elf --list-ptx vadd_ptxonly.fatbin
	@echo "===== launch_overhead 可执行文件里嵌的 GPU 代码 ====="; $(CUDA_HOME)/bin/cuobjdump --list-elf --list-ptx launch_overhead
	@echo "===== vadd 的 SASS ====="; $(CUDA_HOME)/bin/cuobjdump -sass -fun vadd vadd.cubin | grep -E "^\s+/\*[0-9a-f]{4}\*/" | head -30
	@echo "===== 可执行文件里的 .nv_fatbin 段 ====="; readelf -S launch_overhead | grep -A1 -E "nv_fatbin|nvFatBin"

ctx: ctx_explore
	./ctx_explore

overhead: launch_overhead
	@echo "===== LAZY（默认）====="; CUDA_MODULE_LOADING=LAZY ./launch_overhead
	@echo "===== EAGER ====="; CUDA_MODULE_LOADING=EAGER ./launch_overhead | head -5

run-errors: errors
	./errors
procaddr: proc_addr
	./proc_addr
hook: libhook_preview.so drv_vadd launch_overhead $(MODULES)
	@echo "===== 纯 driver 程序 ====="; LD_PRELOAD=./libhook_preview.so ./drv_vadd vadd.cubin > /dev/null
	@echo "===== runtime 程序 ====="; LD_PRELOAD=./libhook_preview.so ./launch_overhead > /dev/null

# 只看 ioctl（和驱动说话的系统调用）。WSL2 上设备文件是 /dev/dxg，原生 Linux 是 /dev/nvidia*
strace: drv_vadd $(MODULES)
	strace -f -c -e trace=ioctl,openat,mmap ./drv_vadd vadd.cubin > /dev/null
	strace -f -e trace=openat ./drv_vadd vadd.cubin 2>&1 >/dev/null | grep -E "dev/|libcuda|ComputeCache" | head -20

nsys: launch_overhead
	nsys profile -t cuda,osrt --stats=true -o l7_overhead -f true ./launch_overhead

clean:
	rm -f $(PROGS) $(MODULES) *.o *.nsys-rep *.sqlite
.PHONY: all run1 jit inspect ctx overhead run-errors procaddr hook strace nsys clean
