# cuda-from-scratch

从零学习 CUDA 的练习代码与实测记录。所有数字均为本人机器实测。

**环境**：NVIDIA GeForce RTX 2050 (2 GB, Laptop)，Windows + WSL2 Ubuntu，驱动 610.47 Studio

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

实测加速比：待补。

```bash
cd lesson3-openmp-mpi
sudo apt install openmpi-bin libopenmpi-dev   # 首次
make && make bench
```

## 编译

```bash
nvcc -O2 -o vector_add lesson1-vector-add/vector_add.cu
nvcc -O2 -o coalesce lesson2-reduce-coalesce/coalesce.cu
nvcc -O2 -o reduce lesson2-reduce-coalesce/reduce.cu
```
