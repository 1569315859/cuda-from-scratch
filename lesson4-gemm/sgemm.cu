// sgemm.cu —— 第 4 课：CUDA 单精度矩阵乘 C = A * B（float，行优先 row-major，N x N）
//
// 版本：
//   v0  CPU 串行 i-k-j（common.h 里的 matmul_ikj_serial），只当「标准答案」
//   v1  sgemm_naive      朴素版：一个线程算一个 C[row][col]，threadIdx.x 对应列（合并访问）
//   v1b sgemm_naive_bad  反面教材：threadIdx.x 对应行（A 的读取不合并），留给 ncu 对比
//   v2  sgemm_tile16 / sgemm_tile32  共享内存分块（tiling）
//   v3  cuBLAS cublasSgemm（编译时加 -DUSE_CUBLAS，Makefile 默认打开），当「天花板」
//
// 用法：
//   ./sgemm [N] [--profile] [--only=naive,naive_bad,tile16,tile32,cublas]
//   N 默认 1024，可以是任意正整数（例如 1000，用来测试边界检查）
//   --profile：每个 kernel 只跑 1 次、不计时（给 ncu / compute-sanitizer 用，避免重复采样）
//   --only=  ：只跑列出的版本，逗号分隔
//
// 故意制造 bug 的编译开关（只用于第 8 章的 compute-sanitizer 演示）：
//   -DSKIP_FIRST_SYNC   去掉分块 kernel 里的第 1 个 __syncthreads()
//   -DSKIP_SECOND_SYNC  去掉分块 kernel 里的第 2 个 __syncthreads()
//   -DNO_BOUNDS         去掉分块 kernel 里加载数据时的边界检查

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"    // fill_random / matmul_ikj_serial / max_rel_err / gflops
#include "common.cuh"  // CUDA_CHECK / CUBLAS_CHECK

#ifndef REF_MAX_N
#define REF_MAX_N 2048  // N 不超过这个值时用 CPU 串行结果当标准答案；更大时 CPU 太慢，改用 cuBLAS
#endif

// ---------------------------------------------------------------------------
// v1：朴素 kernel。线程 (x, y) 负责 C[row][col]，row 来自 y 方向，col 来自 x 方向。
// 16x16 block 里一个 warp = 2 行 x 16 个线程（ty 取 2 个值，tx 取 0..15）：
//   读 A[row][k]：2 个不同地址（每行 16 个线程广播）→ 2 个 sector
//   读 B[k][col]：16 个连续 float（64 字节），两行读同一段 → 2 个 sector；ncu 实测 2.0 sectors/request
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(const float* A, const float* B, float* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[(size_t)row * n + k] * B[(size_t)k * n + col];
        C[(size_t)row * n + col] = sum;
    }
}

// v1b：反面教材。只是把 row 和 col 的来源对调：threadIdx.x 连续 → row 连续。
//   读 A[row][k]：一个 warp = 16 个 tx × 2 个 ty → 碰 16 个不同行、16 个 sector；读 B 只碰 1 个 → ncu 实测 8.5 sectors/request（不是 32）
//   写 C[row][col]：同样跨行，不合并
__global__ void sgemm_naive_bad(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[(size_t)row * n + k] * B[(size_t)k * n + col];
        C[(size_t)row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// v2：共享内存分块。一个 block = TILE x TILE 个线程，负责 C 的一个 TILE x TILE 小块。
// 沿 k 方向分成 numTiles 轮：每轮全 block 合作把 A 的一块、B 的一块搬进共享内存，
// 然后每个线程用共享内存里的数据做 TILE 次乘加。
// ---------------------------------------------------------------------------
template <int TILE>
__device__ __forceinline__ void sgemm_tiled_body(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];  // A 的当前小块
    __shared__ float Bs[TILE][TILE];  // B 的当前小块

    const int tx = threadIdx.x;              // 块内列号
    const int ty = threadIdx.y;              // 块内行号
    const int row = blockIdx.y * TILE + ty;  // 这个线程负责的 C 的行
    const int col = blockIdx.x * TILE + tx;  // 这个线程负责的 C 的列

    float sum = 0.0f;
    const int numTiles = (n + TILE - 1) / TILE;  // 向上取整：N=1000、TILE=32 时是 32 轮

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + tx;  // 本轮要搬的 A[row][aCol]
        const int bRow = t * TILE + ty;  // 本轮要搬的 B[bRow][col]
#ifdef NO_BOUNDS
        As[ty][tx] = A[(size_t)row * n + aCol];   // 故意不检查边界（bug 演示）
        Bs[ty][tx] = B[(size_t)bRow * n + col];
#else
        // 越界的位置填 0：0 参与乘加不改变结果，所以 N 不是 TILE 的倍数也能算对
        As[ty][tx] = (row < n && aCol < n) ? A[(size_t)row * n + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < n && col < n) ? B[(size_t)bRow * n + col] : 0.0f;
#endif

#ifndef SKIP_FIRST_SYNC
        __syncthreads();  // 同步 1：等全 block 都「写完」这一轮的小块，才能开始「读」
#endif
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
#ifndef SKIP_SECOND_SYNC
        __syncthreads();  // 同步 2：等全 block 都「读完」这一轮的小块，才能被下一轮覆盖
#endif
    }

    if (row < n && col < n)  // 只在最后写结果时检查；循环里不能提前 return（见讲义 5.4）
        C[(size_t)row * n + col] = sum;
}

__global__ void sgemm_tile16(const float* A, const float* B, float* C, int n) {
    sgemm_tiled_body<16>(A, B, C, n);
}
__global__ void sgemm_tile32(const float* A, const float* B, float* C, int n) {
    sgemm_tiled_body<32>(A, B, C, n);
}

// ---------------------------------------------------------------------------
// 计时：warmup 次预热（不计时）+ reps 次计时，每次用一对 cudaEvent 量 GPU 上的时间
// ---------------------------------------------------------------------------
struct Timing {
    float avg_ms;
    float min_ms;
};

template <typename Launch>
Timing time_gpu(Launch launch, int warmup, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());       // 检查 kernel 启动参数是否出错
    CUDA_CHECK(cudaDeviceSynchronize());  // 检查 kernel 运行时是否出错

    float total = 0.0f, best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));  // 等 stop 这个「时间戳」真正被 GPU 打上
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {reps > 0 ? total / reps : 0.0f, reps > 0 ? best : 0.0f};
}

static bool wanted(const std::string& only, const char* name) {
    if (only.empty()) return true;
    std::string list = "," + only + ",";
    return list.find("," + std::string(name) + ",") != std::string::npos;
}

int main(int argc, char** argv) {
    int n = 1024;
    bool profile = false;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) profile = true;
        else if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
        else n = std::atoi(argv[i]);
    }
    if (n <= 0) { fprintf(stderr, "N 必须是正整数\n"); return 1; }

    // ---- 1. 设备信息 + 显存检查 ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const size_t nn = (size_t)n * n;
    const size_t bytes = nn * sizeof(float);
    printf("GPU: %s (sm_%d%d), SM 数 %d, 每 block 共享内存上限 %zu KB\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, prop.sharedMemPerBlock / 1024);
    printf("显存: 空闲 %.0f MB / 总共 %.0f MB；本次需要 3 x %.1f MB = %.1f MB\n", freeB / 1048576.0,
           totalB / 1048576.0, bytes / 1048576.0, 3.0 * bytes / 1048576.0);
    if (3 * bytes > freeB * 0.8) {  // 留 20% 余量给 cuBLAS 工作区和驱动
        fprintf(stderr, "N=%d 需要的显存太多，请减小 N\n", n);
        return 1;
    }
    printf("N = %d, 模式 = %s\n\n", n, profile ? "profile（每个 kernel 只跑 1 次，不计时）"
                                              : "benchmark（GPU 唤醒 0.5 秒 + 每版 1 次预热 + 10 次计时）");

    // ---- 2. 主机端数据 ----
    std::vector<float> hA(nn), hB(nn), hC(nn), hRef(nn);
    fill_random(hA, 1);
    fill_random(hB, 2);

    // ---- 3. 设备端数据 ----
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

#ifdef USE_CUBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));  // 创建句柄比较慢，放在计时之外
    const float alpha = 1.0f, beta = 0.0f;
#endif

    // ---- 4. 标准答案 ----
    bool haveRef = false;
    const char* refName = "无";
    if (n <= REF_MAX_N) {
        auto t0 = std::chrono::steady_clock::now();
        matmul_ikj_serial(hA.data(), hB.data(), hRef.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("v0 CPU 串行 i-k-j (float): %.1f ms, %.2f GFLOPS（只跑 1 次，仅供参考）\n", s * 1e3,
               gflops(n, s));
        haveRef = true;
        refName = "CPU i-k-j";
    }
#ifdef USE_CUBLAS
    else {
        // C = A*B（行优先）等价于 C^T = B^T * A^T（列优先），所以把 B 放前面、A 放后面
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n,
                                 &beta, dC, n));
        CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));
        haveRef = true;
        refName = "cuBLAS";
    }
#endif
    printf("正确性标准答案：%s\n\n", refName);

    // ---- 5. 各个 GPU 版本 ----
    const int warmup = profile ? 0 : 1;
    const int reps = profile ? 1 : 10;
    dim3 block16(16, 16);
    dim3 grid16((n + 15) / 16, (n + 15) / 16);
    dim3 block32(32, 32);
    dim3 grid32((n + 31) / 32, (n + 31) / 32);

    // ---- 唤醒 GPU：笔记本独显空闲时会降到低频甚至断电，刚启动时频率还没升上来。
    // 实测（2026-11-23）：冷启动第一次 make run 所有版本都慢 4~5 倍，第二次起才稳定。
    // 所以正式计时前先连续跑朴素 kernel 至少 0.5 秒，把频率「拉满」。profile 模式不需要。
    if (!profile) {
        auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) {
            sgemm_naive<<<grid16, block16>>>(dA, dB, dC, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            ++spins;
        }
        CUDA_CHECK(cudaGetLastError());
        printf("GPU 唤醒：先跑了 %d 次朴素 kernel（约 0.5 秒），不计时\n\n", spins);
    }

    printf("%-16s %10s %10s %10s %12s %s\n", "版本", "平均 ms", "最快 ms", "GFLOPS", "max_rel_err",
           "结果");

    auto run = [&](const char* name, auto launch) {
        if (!wanted(only, name)) return;
        CUDA_CHECK(cudaMemset(dC, 0, bytes));  // 清零，防止上一个版本的结果「冒充」本版本
        Timing t = time_gpu(launch, warmup, reps);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double err = haveRef ? max_rel_err(hC.data(), hRef.data(), nn) : -1.0;
        const char* verdict = !haveRef ? "未检查" : (err < 1e-3 ? "OK" : "WRONG");
        if (profile)
            printf("%-16s %10s %10s %10s %12.2e %s\n", name, "-", "-", "-", err, verdict);
        else
            printf("%-16s %10.3f %10.3f %10.1f %12.2e %s\n", name, t.avg_ms, t.min_ms,
                   gflops(n, t.avg_ms / 1e3), err, verdict);
    };

    run("naive", [&] { sgemm_naive<<<grid16, block16>>>(dA, dB, dC, n); });
    run("naive_bad", [&] { sgemm_naive_bad<<<grid16, block16>>>(dA, dB, dC, n); });
    run("tile16", [&] { sgemm_tile16<<<grid16, block16>>>(dA, dB, dC, n); });
    run("tile32", [&] { sgemm_tile32<<<grid32, block32>>>(dA, dB, dC, n); });
#ifdef USE_CUBLAS
    run("cublas", [&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n,
                                 &beta, dC, n));
    });
    CUBLAS_CHECK(cublasDestroy(handle));
#endif

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    printf("\nGFLOPS 按平均时间计算 = 2*N^3 / 平均时间。\n");
    return 0;
}
