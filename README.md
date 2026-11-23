# cuda-from-scratch

从零学习 CUDA 的练习代码与实测记录。所有数字均为本人机器实测。

**环境**：NVIDIA GeForce RTX 2050 (2 GB, Laptop)，Windows + WSL2 Ubuntu，驱动 610.47 Studio

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

实测结果：待补。

```bash
cd lesson4-gemm
make run            # 编译并运行
make ncu-basic      # Nsight Compute 分析（需先在 Windows 端开启 GPU 性能计数器）
make racecheck      # compute-sanitizer 竞争检查
make cachegrind     # CPU 缓存分析
```

## 编译

```bash
nvcc -O2 -o vector_add lesson1-vector-add/vector_add.cu
nvcc -O2 -o coalesce lesson2-reduce-coalesce/coalesce.cu
nvcc -O2 -o reduce lesson2-reduce-coalesce/reduce.cu
```
