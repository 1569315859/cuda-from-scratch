# cuda-from-scratch

从零学习 CUDA 的练习代码与实测记录。所有数字均为本人机器实测。

**环境**：NVIDIA GeForce RTX 2050 (4 GB, Laptop)，Windows + WSL2 Ubuntu，驱动 620.73，CUDA 13.5，Nsight Compute 2026.4

每课目录下附有讲义 PDF（`CUDA第N课详细版.pdf`）。

**CPU**：Intel i7-12700H，6 个性能核（带超线程）+ 8 个能效核 = 14 核 20 线程，L3 24 MB。注意 WSL 里 `lscpu` 显示为「10 核 × 2 线程」，与真实拓扑不符。

## 第 1 课 · 向量加法（`lesson1-vector-add/`）

| 项目 | 实测 |
|---|---|
| kernel（预热后） | 1.914 ms，105.2 GB/s |
| H2D 普通内存 / 锁页内存 | 66.62 ms / 34.86 ms |
| D2H 普通内存 / 锁页内存 | 25.52 ms / 10.92 ms |

结论：kernel 本身很快，瓶颈在 PCIe 传输；锁页内存（pinned）让拷贝快约 2 倍。

## 第 2 课 · 归约与合并访问（`lesson2-reduce-coalesce/`）

`coalesce.cu`：拷贝 64 MB，一个 warp 的 32 个线程被分成 S 组访问不同区域。

| S | 时间 | 带宽 |
|---|---|---|
| 1 | 1.275 ms | 105.2 GB/s |
| 2 | 1.282 ms | 104.7 GB/s |
| 4 | 1.278 ms | 105.0 GB/s |
| 8 | 1.263 ms | 106.3 GB/s |
| 16 | 1.303 ms | 103.0 GB/s |
| 32 | 7.029 ms | 19.1 GB/s |

观察：S≤16 几乎不掉，S=32 断崖式下降约 5.5 倍。推测原因：显存按 32 字节 sector 搬运，相邻 warp 很快用到同一 sector 剩余数据，L2 掩盖了轻度不合并；S=32 时每条访存指令触达 32 个相距 2 MB 的 sector。待用 Nsight Compute 验证。

`reduce.cu`：5 个版本的求和（atomic → 交错归约 → 顺序归约 → 加载时相加 → grid-stride + warp shuffle）。实测结果待补。

## 第 3 课 · OpenMP / MPI 矩阵乘法（`lesson3-openmp-mpi/`）

- `omp_sum.cpp`：竞争条件、critical / atomic / reduction、伪共享对比
- `omp_matmul.cpp`：i-j-k vs i-k-j 循环顺序，OpenMP 并行，GFLOPS = 2N³/t
- `mpi_matmul.cpp`：按行分块（Scatter / Bcast / Gather），分发、计算、回收分别计时

实测（N = 1024 双精度，`make bench`，单次运行，2026-11-23）：

**OpenMP 矩阵乘法（i-k-j）**，加速比相对同一次运行的串行 i-k-j（串行 551–625 ms，波动约 13%）：

| 线程 | 时间 | GFLOPS | 加速比 |
|---|---|---|---|
| 1 | 589.2 ms | 3.64 | 1.06× |
| 2 | 391.8 ms | 5.48 | 1.48× |
| 4 | 160.2 ms | 13.40 | 3.78× |
| 6 | 136.3 ms | 15.76 | 4.04× |
| 8 | 111.9 ms | 19.20 | 5.10× |
| 12 | 102.6 ms | 20.94 | 5.55× |
| **14** | **95.2 ms** | **22.56** | **6.37×** |
| 20 | 118.4 ms | 18.14 | 5.01× |

**MPI 矩阵乘法（按行分块）**，加速比相对 1 进程（556.7 ms）：

| 进程 | 总时间 | GFLOPS | 加速比 | 通信占比 |
|---|---|---|---|---|
| 1 | 556.7 ms | 3.86 | 1.00× | 0.2% |
| 2 | 283.0 ms | 7.59 | 1.97× | 0.9% |
| 4 | 196.2 ms | 10.94 | 2.84× | 2.1% |
| 6 | 171.5 ms | 12.52 | 3.25× | 3.6% |
| 8 | 179.2 ms | 11.98 | 3.11× | 3.6% |
| 12 | 141.3 ms | 15.20 | 3.94× | 5.5% |
| **14** | **131.7 ms** | **16.31** | **4.23×** | 6.7% |
| 20 | 241.6 ms | 8.89 | 2.30× | 分发 73.6 ms |

**OpenMP 求和（`omp_sum`，3200 万个 double，reduction 版，三次平均）**：1 线程 28.3 ms → 6 线程 8.75 ms（3.2×，约 37 GB/s 峰值）→ 20 线程 13.25 ms（2.1×）。

观察：
- 矩阵乘法是计算受限，一直涨到 14 线程（= 物理核数）；求和是访存受限，6 线程就把内存带宽用满。
- 两者开到 20 都变慢：多出的 6 个是超线程，抢同一套计算单元。
- 并行效率随线程数下降，主要原因是大小核速度不同，而 `schedule(static)` 平均分活（MPI 8 进程时最慢与最快进程计算时间差 33%）。
- 单机上 OpenMP 比 MPI 快（95.2 vs 131.7 ms）：线程共享一份 B 矩阵，MPI 每个进程要各拷贝一份（8 MB）再加通信。
- 待改进：伪共享演示在 -O2 下效果不稳定（编译器可能把累加放在寄存器），计划改用 volatile 重做；尝试 `schedule(dynamic)` 缓解大小核不均。

```bash
cd lesson3-openmp-mpi
sudo apt install openmpi-bin libopenmpi-dev   # 首次
make && make bench COUNTS="1 2 4 6 8 12 14 20"
```

## 第 4 课 · CUDA 矩阵乘法 SGEMM（`lesson4-gemm/`）

- `sgemm.cu`：v1 朴素 kernel（每线程算一个 C 元素）→ v2 shared memory 分块（TILE=16 / 32）→ v3 cuBLAS 作为上限；cudaEvent 计时（1 次预热 + 10 次），GFLOPS = 2N³/t，与 CPU 结果对比误差
- 另含反面教材 `sgemm_naive_bad`（不合并访问）供 Nsight Compute 对比；编译开关 `SKIP_FIRST_SYNC` / `SKIP_SECOND_SYNC` / `NO_BOUNDS` 故意制造 bug，供 compute-sanitizer 检查
- `cache_order.cpp`：i-j-k vs i-k-j，用 cachegrind 看 CPU 缓存缺失率

实测结果（RTX 2050 4 GB 笔记本，WSL2，float，GPU 已唤醒后的稳定结果，每版 1 次预热 + 10 次取平均，2026-11-23，多次复跑一致）：

| 版本 | N=1024 平均 ms | GFLOPS | N=1000 平均 ms | GFLOPS | 相对 naive（N=1024） |
|---|---|---|---|---|---|
| naive | 6.802 | 315.7 | 5.155 | 388.0 | 1.00× |
| naive_bad（不合并） | 24.797 | 86.6 | 10.916 | 183.2 | 0.27× |
| tile16 | 4.070 | 527.6 | 4.129 | 484.3 | 1.67× |
| tile32 | 4.859 | 441.9 | 4.791 | 417.5 | 1.40× |
| cuBLAS | 0.507 | 4236.5 | 0.508 | 3934.4 | 13.4× |

规模扫描（GFLOPS，同一天复跑，GPU 已唤醒）：

| N | naive | naive_bad | tile16 | tile32 | cuBLAS |
|---|---|---|---|---|---|
| 256 | 294.2 | 73.3 | 372.4 | 369.0 | 1400.3 |
| 512 | 311.6 | 73.2 | 404.5 | 368.1 | 2318.7 |
| 1024 | 315.6 | 97.4 | 528.8 | 411.7 | 4227.7 |
| 2048 | 325.2 | 102.1 | 419.3 | 480.0 | 3758.4 |
| 4096 | 321.8 | 86.8 | 351.3 | 426.1 | 2187.8 |

- N ≤ 1024 时 tile16 更快，N ≥ 2048 时 tile32 反超。N=1024 的 ncu 证实了两边的原因：tile32 显存读取减半（142.83 MB vs 274.20 MB），但理论/实测占用率只有 66.67% / 66.63%（tile16 约 99%）。N=2048 的 ncu 还没测
- N=256 时工作量太小，GPU 喂不饱，cuBLAS 也只有 1400 GFLOPS
- N=4096 时 cuBLAS 和 tile16 的平均/最快时间差距变大，疑似长时间满载后笔记本降频；N=4096 正确性参照为 cuBLAS（CPU 太慢跳过）

- 所有版本与 CPU 串行结果对比 max_rel_err ≤ 2.3e-6（cuBLAS 求和顺序不同，误差略大）
- `-Xptxas -v`：四个 kernel 均 0 spill；tile32 用 38 寄存器 / 8 KB smem，按寄存器换算每 SM 只能放 1 个 1024 线程 block（理论占用率 67%），tile16 理论 100%，与 tile16 更快一致
- 手写分块与 cuBLAS 仍差约 8×，第 5 课用寄存器分块缩小差距
- 踩坑：编译完第一次运行时 GPU 各版本（含 cuBLAS）都慢 4.0~5.4 倍（cuBLAS 仅 787 GFLOPS），CPU 只慢约 1.5 倍，第二次起稳定在上表。判断是笔记本独显空闲降频、1 次预热太短没把频率拉起来。已在 `sgemm.cu` 里加入正式计时前「连续跑 0.5 秒」的 GPU 唤醒步骤

**Nsight Compute（N=1024，2026-11-24；ncu 把 SM 锁在 1.24 GHz，所以时间只看比例）**：

| 指标 | naive | naive_bad | tile16 | tile32 |
|---|---|---|---|---|
| 全局读 sectors/request | 2.0 | 8.5 | 3.98 | 4.0 |
| sectors 总数 | 134,217,432 | 570,424,428 | 16,700,520 | 8,388,608 |
| DRAM 读取 | 274.25 MB | 276.94 MB | 274.20 MB | 142.83 MB |
| L1/TEX 吞吐 | 99.33% | 97.78% | 97.1% | 约 79% |
| DRAM 吞吐 | 36.67% | 8.54% | 47.85% | 约 22.5% |
| Mem Pipes Busy | 99.25% | 22.99% | 96.99% | 78.68% |
| 实测占用率 | 99.01% | 99.13% | 99.03% | 66.63% |

- naive 卡在 L1 和访存指令，不是显存：L1/TEX 99%，DRAM 只有 37%
- naive_bad：16×16 block 里一个 warp 是 2 行 × 16 线程，读 A 碰 16 个 sector、读 B 碰 1 个，(16+1)/2 = 8.5（不是 32）。sectors 是 naive 的 4.25 倍，耗时约 4.3 倍，显存字节却几乎一样，代价在 L1 处理 sector
- tile16 显存字节和 naive 一样，快在全局读请求少了 16 倍；共享内存和 L1 是同一块硬件，所以 L1/TEX 仍是 97%
- 所有 kernel 的 DRAM 吞吐都不超过 48%，真正的墙是访存指令太多 → 第 5 课寄存器分块

**compute-sanitizer（2026-11-24）**：
- memcheck 正常版 N=1000：0 errors；去掉边界检查（`sgemm_nobounds`）：报 `Invalid __global__ read`，指向 sgemm.cu:90（最后一轮读 B，bRow = 1002），地址与手算下标 1,002,288 一致
- racecheck 正常版 N=256：0 hazards；删掉第二个 `__syncthreads()`（`sgemm_nosync`）：报 2 errors（第 94 行写 Bs ↔ 第 101 行读）、3 warnings（第 93 行写 As ↔ 第 101 行读，及记录上限提示），跑了两次，tile16 / tile32 都 WRONG，误差两次不同（tile16 9.15e-2 → 1.01e-1，tile32 1.12e-1 → 8.87e-2），竞争导致的结果不确定
- 待做：cachegrind（WSL2 上 perf 读不到硬件计数器，`cycles event is not supported`）

```bash
cd lesson4-gemm
make run            # 编译并运行
make ncu-basic      # Nsight Compute 分析（需先在 Windows 端开启 GPU 性能计数器）
make racecheck      # compute-sanitizer 竞争检查
make cachegrind     # CPU 缓存分析
```

## 第 5 课 · 寄存器分块（`lesson5-register-blocking/`）

- `sgemm5.cu`：tile16（第 4 课基线）→ 1D 寄存器分块（每线程算 8 个 C）→ 2D 寄存器分块 8×8（外积）→ 2D + float4 向量化读写
- 目标：减少共享内存读指令，把第 4 课测到的 LSU 瓶颈移走
- `-Xptxas -v` 已实测（2026-11-25）：tile16 37 寄存器 / 2 KB smem，1d_tm8 48 / 4 KB，2d_8x8 128 / 8 KB，2d_vec 117 / 8 KB，全部 0 spill
- 实测（2026-11-25，N = 1024）：tile16 472.4 → 1d_tm8 1372.9 → 2d_8x8 2466.6 → 2d_vec **3118.1 GFLOPS**（tile16 的 6.60 倍，cuBLAS 4230.6 的 73.7%）；共享读指令 41.9M → 2.1M，FFMA 恒为 N³，Mem Pipes Busy 97% → 31%
- 预测失误：A 转置没有减少共享读 wavefront（两者都是每条 LDS.128 5 个）；按半 warp 分相的模型（推测）能精确解释四个 kernel，冲突全在读 B 一侧。详见目录下 README.md

```bash
cd lesson5-register-blocking
make run && make ncu-sol
```

## 第 6 课 · 双缓冲、cp.async、warp 分块、Tensor Core（`lesson6-double-buffer-tensorcore/`）

GEMM 主线收尾，下一课转向 GPGPU Runtime（驱动 API / 上下文 / 模块 / kernel 启动路径）。

- `sgemm6.cu`：2d_vec（第 5 课基线）→ db_reg（寄存器预取双缓冲，每轮 1 次同步）→ db_cpasync（cp.async 两级流水线，全局→共享不经过寄存器）→ warptile（block 128×128 → warp 64×32 → 线程 2×2 个 4×4）→ wmma_naive / wmma_fp16（Tensor Core，FP16 输入 FP32 累加）；对照 cuBLAS FP32 / TF32 / FP16
- 编译开关 `SKIP_CP_WAIT`（去掉 cp.async wait）/ `SKIP_DB_SYNC`（去掉双缓冲唯一的同步）供 compute-sanitizer 演示；`test_l6_emulation.cpp` 在无 GPU 机器上用 CPU 线程模拟检查下标与流水线顺序（`make emu`）
- 编译器参考（nvcc 13.5，sm_86，`-Xptxas -v`）：2d_vec 117 / db_reg 128（不加 `__launch_bounds__(256, 2)` 时 129，每 SM 只剩 1 个 block）/ db_cpasync 117 / warptile 119 / wmma_naive 40 / wmma_fp16 122 寄存器，全部 0 spill
- 实测结果待补（见目录下 README.md）

```bash
cd lesson6-double-buffer-tensorcore
make ptxas && make run && make ncu-metrics
```

## 第 7 课 · Driver API、Context、Module 与 kernel 启动路径（`lesson7-driver-api-launch/`）

Runtime 方向第一课：一行 `<<<>>>` 从编译器改写、runtime 注册表、driver 提交到 GPU 分发的全过程。

- `drv_vadd.cpp`：只用 driver API 跑向量加法（`cuInit` → primary context → `cuModuleLoad` / `cuLibraryLoadFromFile` → `cuLaunchKernel`），分别加载 cubin / PTX / fatbin / 纯 PTX fatbin（JIT）
- `ctx_explore.cu`：runtime 与 driver 共用 primary context、线程局部的 current context、自建 context 的耗时与显存
- `launch_overhead.cu`：懒加载 vs 立即加载的首次启动、异步提交与队列反压、启动 + 同步来回、`<<<>>>` / `cudaLaunchKernel` / `cuLaunchKernel` 对比
- `errors.cu`：非粘性 vs 粘性错误，`cudaDeviceReset` 恢复
- `linux/`：预备课，Linux 基础从零讲（线程局部存储、虚拟内存与 mmap、系统调用、动态库与 ELF、dlopen 与 LD_PRELOAD），不需要 GPU
- `proc_addr.cu`、`hook_preview.cpp`：`dlsym` vs `cuGetProcAddress`，LD_PRELOAD 拦截 `cuLaunchKernel`（第 10 课预告）
- 沙箱验证（nvcc 13.5）：静态链接的 runtime 导入表里没有 driver 函数，运行时通过 `cuGetProcAddress` 取地址；实测结果待补（见目录下 README.md）

```bash
cd lesson7-driver-api-launch
make && make run1 && make overhead && make hook
```

## 编译

```bash
nvcc -O2 -o vector_add lesson1-vector-add/vector_add.cu
nvcc -O2 -o coalesce lesson2-reduce-coalesce/coalesce.cu
nvcc -O2 -o reduce lesson2-reduce-coalesce/reduce.cu
```
