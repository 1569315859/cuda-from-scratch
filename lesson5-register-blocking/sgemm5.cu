// sgemm5.cu —— 第 5 课：寄存器分块（register blocking）/ 线程粗化（thread coarsening）
// C = A * B，float，行优先（row-major），N x N
//
// 版本：
//   sgemm_tile16  第 4 课的共享内存分块（原样复制），当「基线」：每个线程算 1 个 C
//   sgemm_1d_tm8  一维寄存器分块：block 负责 64x64 的 C，512 个线程，每个线程算一列上的 8 个 C
//   sgemm_2d_8x8  二维寄存器分块：block 负责 128x128 的 C，256 个线程，每个线程算 8x8=64 个 C
//   sgemm_2d_vec  在 2d_8x8 基础上：float4 向量化读写全局内存 + 共享内存里把 A 小块转置存放
//   cublas        cuBLAS cublasSgemm（编译时加 -DUSE_CUBLAS，Makefile 默认打开），当「天花板」
//
// 用法：
//   ./sgemm5 [N] [--profile] [--only=tile16,1d_tm8,2d_8x8,2d_vec,cublas]
//   N 默认 1024。1d_tm8 要求 N 是 64 的倍数，2d_8x8 / 2d_vec 要求 N 是 128 的倍数，否则自动跳过
//   --profile：每个 kernel 只跑 1 次、不计时（给 ncu / compute-sanitizer 用）
//   --only=  ：只跑列出的版本，逗号分隔
//
// 故意制造 bug 的编译开关（只用于讲义第 9、10 章的演示）：
//   -DSKIP_FIRST_SYNC   去掉所有分块 kernel 里「搬完数据」后的 __syncthreads()
//   -DSKIP_SECOND_SYNC  去掉所有分块 kernel 里「算完这一轮」后的 __syncthreads()
//   -DNO_BOUNDS         去掉 tile16 搬数据时的边界检查（同第 4 课）
//   -DNO_N_CHECK        不检查 N 是不是 64 / 128 的倍数，硬跑寄存器分块 kernel（会越界）

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
#define REF_MAX_N 2048  // N 不超过这个值时用 CPU 串行结果当标准答案；更大时改用 cuBLAS
#endif

// 两个同步点写成宏，方便用编译开关「故意删掉」它们
#ifdef SKIP_FIRST_SYNC
#define SYNC_AFTER_LOAD()      // bug 演示：搬完数据不等别人
#else
#define SYNC_AFTER_LOAD() __syncthreads()
#endif
#ifdef SKIP_SECOND_SYNC
#define SYNC_AFTER_COMPUTE()   // bug 演示：算完不等别人就进入下一轮
#else
#define SYNC_AFTER_COMPUTE() __syncthreads()
#endif

// ======== GPU kernels 开始 ========

// ---------------------------------------------------------------------------
// 基线：第 4 课的 tile16（一个线程算 1 个 C，内层循环每次 FMA 读 2 次共享内存）
// ---------------------------------------------------------------------------
__global__ void sgemm_tile16(const float* A, const float* B, float* C, int n) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    const int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;
#ifdef NO_BOUNDS
        As[ty][tx] = A[(size_t)row * n + aCol];
        Bs[ty][tx] = B[(size_t)bRow * n + col];
#else
        As[ty][tx] = (row < n && aCol < n) ? A[(size_t)row * n + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < n && col < n) ? B[(size_t)bRow * n + col] : 0.0f;
#endif
        SYNC_AFTER_LOAD();
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        SYNC_AFTER_COMPUTE();
    }
    if (row < n && col < n)
        C[(size_t)row * n + col] = sum;
}

// ---------------------------------------------------------------------------
// v1：一维寄存器分块（1D blocktiling）
//   一个 block 负责 C 的 64 x 64 小块；沿 k 方向每轮搬 A 的 64x8、B 的 8x64 进共享内存。
//   512 个线程，排成 8 行 x 64 列：线程 (threadRow, threadCol) 负责
//   C 小块里第 threadRow*8 ... threadRow*8+7 这 8 行、第 threadCol 列 —— 竖着的一串 8 个数。
//   每读 1 个 B 的数（放进寄存器 b），可以和 8 个 A 的数各做一次乘加。
// ---------------------------------------------------------------------------
namespace cfg1d {
constexpr int BM = 64;                     // block 负责的 C 小块高度（行数）
constexpr int BN = 64;                     // block 负责的 C 小块宽度（列数）
constexpr int BK = 8;                      // 每轮沿 k 方向搬多宽
constexpr int TM = 8;                      // 每个线程算竖着的 8 个 C
constexpr int THREADS = BM * BN / TM;      // 64*64/8 = 512 个线程
}  // namespace cfg1d

__global__ void __launch_bounds__(cfg1d::THREADS)
sgemm_1d_tm8(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int n) {
    using namespace cfg1d;
    __shared__ float As[BM * BK];          // 64 行 x 8 列，As[r*BK + c]
    __shared__ float Bs[BK * BN];          // 8 行 x 64 列，Bs[r*BN + c]

    const int cRow = blockIdx.y;           // 第几个 C 小块（纵向）
    const int cCol = blockIdx.x;           // 第几个 C 小块（横向）
    const int tid = threadIdx.x;           // 0..511，一维 block

    // 计算阶段的分工：8 行 x 64 列个线程
    const int threadCol = tid % BN;        // 0..63
    const int threadRow = tid / BN;        // 0..7

    // 搬运阶段的分工：A 小块 64x8=512 个数、B 小块 8x64=512 个数，每个线程各搬 1 个
    const int innerRowA = tid / BK;        // 0..63
    const int innerColA = tid % BK;        // 0..7
    const int innerRowB = tid / BN;        // 0..7
    const int innerColB = tid % BN;        // 0..63

    const float* Ablk = A + (size_t)cRow * BM * n;   // A 的行带起点：第 cRow*64 行
    const float* Bblk = B + (size_t)cCol * BN;       // B 的列带起点：第 cCol*64 列

    float acc[TM] = {0.0f};                // 8 个累加器，放在寄存器里

    for (int bk = 0; bk < n; bk += BK) {   // 沿 k 方向，每轮前进 8
        As[innerRowA * BK + innerColA] = Ablk[(size_t)innerRowA * n + bk + innerColA];
        Bs[innerRowB * BN + innerColB] = Bblk[(size_t)(bk + innerRowB) * n + innerColB];
        SYNC_AFTER_LOAD();

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            const float b = Bs[dot * BN + threadCol];        // 1 次共享内存读 → 寄存器
#pragma unroll
            for (int i = 0; i < TM; ++i)                     // 8 次乘加都用这个 b
                acc[i] += As[(threadRow * TM + i) * BK + dot] * b;
        }
        SYNC_AFTER_COMPUTE();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
        C[(size_t)(cRow * BM + threadRow * TM + i) * n + cCol * BN + threadCol] = acc[i];
}

// ---------------------------------------------------------------------------
// v2：二维寄存器分块（2D blocktiling）
//   一个 block 负责 C 的 128 x 128 小块；每轮搬 A 的 128x8、B 的 8x128 进共享内存。
//   256 个线程，排成 16 x 16：线程 (threadRow, threadCol) 负责 C 小块里
//   第 threadRow*8 .. +7 行、第 threadCol*8 .. +7 列 —— 一个 8x8 的小方块，64 个数。
//   每个 dot：从共享内存读 8 个 A（一小列）进 regM、8 个 B（一小行）进 regN，做 8x8=64 次乘加。
// ---------------------------------------------------------------------------
namespace cfg2d {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);   // 16*16 = 256
}  // namespace cfg2d

__global__ void __launch_bounds__(cfg2d::THREADS)
sgemm_2d_8x8(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int n) {
    using namespace cfg2d;
    __shared__ float As[BM * BK];          // 128 行 x 8 列，As[r*BK + c]
    __shared__ float Bs[BK * BN];          // 8 行 x 128 列，Bs[r*BN + c]

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.x;           // 0..255

    const int threadCol = tid % (BN / TN); // 0..15
    const int threadRow = tid / (BN / TN); // 0..15

    // 搬运分工：A 小块 128x8 = 1024 个数，256 个线程 → 每人搬 4 个（相隔 32 行）
    const int innerRowA = tid / BK;        // 0..31
    const int innerColA = tid % BK;        // 0..7
    constexpr int strideA = THREADS / BK;  // 32：全体线程一次能搬 32 行
    // B 小块 8x128 = 1024 个数 → 每人搬 4 个（相隔 2 行）
    const int innerRowB = tid / BN;        // 0..1
    const int innerColB = tid % BN;        // 0..127
    constexpr int strideB = THREADS / BN;  // 2：全体线程一次能搬 2 行

    const float* Ablk = A + (size_t)cRow * BM * n;
    const float* Bblk = B + (size_t)cCol * BN;

    float acc[TM * TN] = {0.0f};           // 64 个累加器
    float regM[TM];                        // 寄存器缓存：A 的一小列（8 个）
    float regN[TN];                        // 寄存器缓存：B 的一小行（8 个）

    for (int bk = 0; bk < n; bk += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] =
                Ablk[(size_t)(innerRowA + off) * n + bk + innerColA];
#pragma unroll
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] =
                Bblk[(size_t)(bk + innerRowB + off) * n + innerColB];
        SYNC_AFTER_LOAD();

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dot];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[dot * BN + threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        SYNC_AFTER_COMPUTE();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(size_t)(cRow * BM + threadRow * TM + i) * n
              + cCol * BN + threadCol * TN + j] = acc[i * TN + j];
}

// ---------------------------------------------------------------------------
// v3：二维寄存器分块 + float4 向量化 + A 小块转置
//   和 v2 分工完全一样，只改「怎么搬」：
//   (1) 全局内存一次读 / 写 4 个连续 float（float4，16 字节），读写指令数变成 1/4
//   (2) A 小块在共享内存里转置成 As[k][m]（8 行 x 128 列），这样计算时一个线程要的
//       8 个 A（同一个 k、连续 8 个 m）在共享内存里是连续的，能用 2 次 float4 读完
//   float4 要求地址是 16 字节的倍数：cudaMalloc 返回的地址至少 256 字节对齐，
//   N 是 128 的倍数 → 每行起点、每个 bk 起点都是 16 字节的倍数；共享数组用 __align__(16)。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(cfg2d::THREADS)
sgemm_2d_vec(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int n) {
    using namespace cfg2d;
    __shared__ __align__(16) float As[BK * BM];   // 转置后 8 行(k) x 128 列(m)：As[k*BM+m]
    __shared__ __align__(16) float Bs[BK * BN];   // 8 行(k) x 128 列(n)，Bs[k*BN + c]

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.x;

    const int threadCol = tid % (BN / TN); // 0..15
    const int threadRow = tid / (BN / TN); // 0..15

    // 搬 A：128 行 x 8 列 = 每行 2 个 float4，共 256 个 float4 → 每个线程正好 1 个
    const int innerRowA = tid / (BK / 4);  // 0..127
    const int innerColA = tid % (BK / 4);  // 0..1（第几个 float4）
    // 搬 B：8 行 x 128 列 = 每行 32 个 float4，共 256 个 float4 → 每个线程正好 1 个
    const int innerRowB = tid / (BN / 4);  // 0..7
    const int innerColB = tid % (BN / 4);  // 0..31

    const float* Ablk = A + (size_t)cRow * BM * n;
    const float* Bblk = B + (size_t)cCol * BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < n; bk += BK) {
        // A：一次读 4 个连续的 A[row][bk+4c .. bk+4c+3]，拆开转置写进 As 的 4 行
        const float4 a4 = *reinterpret_cast<const float4*>(
            &Ablk[(size_t)innerRowA * n + bk + innerColA * 4]);
        As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
        // B：不用转置，float4 原样搬
        *reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4]) =
            *reinterpret_cast<const float4*>(
                &Bblk[(size_t)(bk + innerRowB) * n + innerColB * 4]);
        SYNC_AFTER_LOAD();

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            // 8 个 A：As[dot][threadRow*8 .. +7]，连续 → 2 次 float4 读
            const float4* aRow = reinterpret_cast<const float4*>(&As[dot * BM + threadRow * TM]);
            const float4 m0 = aRow[0];
            const float4 m1 = aRow[1];
            // 8 个 B：Bs[dot][threadCol*8 .. +7]，连续 → 2 次 float4 读
            const float4* bRow = reinterpret_cast<const float4*>(&Bs[dot * BN + threadCol * TN]);
            const float4 n0 = bRow[0];
            const float4 n1 = bRow[1];
            regM[0] = m0.x; regM[1] = m0.y; regM[2] = m0.z; regM[3] = m0.w;
            regM[4] = m1.x; regM[5] = m1.y; regM[6] = m1.z; regM[7] = m1.w;
            regN[0] = n0.x; regN[1] = n0.y; regN[2] = n0.z; regN[3] = n0.w;
            regN[4] = n1.x; regN[5] = n1.y; regN[6] = n1.z; regN[7] = n1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        SYNC_AFTER_COMPUTE();
    }

    // 写 C：每行 8 个连续的数 → 2 次 float4 写
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float* cPtr =
            &C[(size_t)(cRow * BM + threadRow * TM + i) * n + cCol * BN + threadCol * TN];
        reinterpret_cast<float4*>(cPtr)[0] =
            make_float4(acc[i * TN + 0], acc[i * TN + 1], acc[i * TN + 2], acc[i * TN + 3]);
        reinterpret_cast<float4*>(cPtr)[1] =
            make_float4(acc[i * TN + 4], acc[i * TN + 5], acc[i * TN + 6], acc[i * TN + 7]);
    }
}

// ======== GPU kernels 结束 ========

// ---------------------------------------------------------------------------
// 计时：warmup 次预热（不计时）+ reps 次计时（和第 4 课完全一样）
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.0f, best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {reps > 0 ? total / reps : 0.0f, reps > 0 ? best : 0.0f};
}

static bool wanted(const std::string& only, const char* name) {
    if (only.empty()) return true;
    std::string list = "," + only + ",";
    return list.find("," + std::string(name) + ",") != std::string::npos;
}

// 打印一个 kernel 的「资源账单」：每线程寄存器、静态共享内存、溢出到 local memory 的字节，
// 以及按这些资源算出来的每个 SM 最多能同时放几个 block、理论占用率
template <typename Kernel>
static void print_resources(const char* name, Kernel kernel, int threadsPerBlock,
                            const cudaDeviceProp& prop) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, threadsPerBlock, 0));
    const double occ = 100.0 * blocksPerSM * threadsPerBlock / prop.maxThreadsPerMultiProcessor;
    printf("%-10s %8d %10d %12zu %14zu %12d %10.1f%%\n", name, threadsPerBlock, attr.numRegs,
           attr.sharedSizeBytes, attr.localSizeBytes, blocksPerSM, occ);
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
    printf("GPU: %s (sm_%d%d), SM 数 %d, 每 SM 寄存器 %d 个, 每 SM 最多 %d 线程\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, prop.regsPerMultiprocessor,
           prop.maxThreadsPerMultiProcessor);
    printf("显存: 空闲 %.0f MB / 总共 %.0f MB；本次需要 3 x %.1f MB = %.1f MB\n", freeB / 1048576.0,
           totalB / 1048576.0, bytes / 1048576.0, 3.0 * bytes / 1048576.0);
    if (3 * bytes > freeB * 0.8) {
        fprintf(stderr, "N=%d 需要的显存太多，请减小 N\n", n);
        return 1;
    }
    printf("N = %d, 模式 = %s\n\n", n, profile ? "profile（每个 kernel 只跑 1 次，不计时）"
                                              : "benchmark（GPU 唤醒 0.5 秒 + 每版 1 次预热 + 10 次计时）");

    // ---- 2. 资源账单（编译器实际分配的寄存器数，运行时查询，不用猜）----
    if (!profile) {
        printf("%-10s %8s %10s %12s %14s %12s %11s\n", "kernel", "线程/blk", "寄存器/线程",
               "静态smem(B)", "local溢出(B)", "blocks/SM", "理论占用率");
        print_resources("tile16", sgemm_tile16, 256, prop);
        print_resources("1d_tm8", sgemm_1d_tm8, cfg1d::THREADS, prop);
        print_resources("2d_8x8", sgemm_2d_8x8, cfg2d::THREADS, prop);
        print_resources("2d_vec", sgemm_2d_vec, cfg2d::THREADS, prop);
        printf("\n");
    }

    // ---- 3. 主机端数据 ----
    std::vector<float> hA(nn), hB(nn), hC(nn), hRef(nn);
    fill_random(hA, 1);
    fill_random(hB, 2);

    // ---- 4. 设备端数据（cudaMalloc 返回的地址至少 256 字节对齐，float4 读写放心用）----
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

#ifdef USE_CUBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
#endif

    // ---- 5. 标准答案 ----
    bool haveRef = false;
    const char* refName = "无";
    if (n <= REF_MAX_N) {
        auto t0 = std::chrono::steady_clock::now();
        matmul_ikj_serial(hA.data(), hB.data(), hRef.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("CPU 串行 i-k-j (float): %.1f ms（只跑 1 次，仅用来当标准答案）\n", s * 1e3);
        haveRef = true;
        refName = "CPU i-k-j";
    }
#ifdef USE_CUBLAS
    else {
        // 行优先技巧（第 4 课 5.2）：C = A*B 等价于 C^T = B^T * A^T（列优先）
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n,
                                 &beta, dC, n));
        CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));
        haveRef = true;
        refName = "cuBLAS";
    }
#endif
    printf("正确性标准答案：%s\n\n", refName);

    // ---- 6. 启动配置 ----
    const int warmup = profile ? 0 : 1;
    const int reps = profile ? 1 : 10;
    dim3 block16(16, 16);
    dim3 grid16((n + 15) / 16, (n + 15) / 16);
    dim3 block1d(cfg1d::THREADS);                                         // 512 个线程，一维
    dim3 grid1d((n + cfg1d::BN - 1) / cfg1d::BN, (n + cfg1d::BM - 1) / cfg1d::BM);  // x 管列、y 管行
    dim3 block2d(cfg2d::THREADS);                                         // 256 个线程，一维
    dim3 grid2d((n + cfg2d::BN - 1) / cfg2d::BN, (n + cfg2d::BM - 1) / cfg2d::BM);

    // ---- 唤醒 GPU（第 4 课实测：冷启动第一次所有版本都慢 4~5 倍）：先连续跑 tile16 至少 0.5 秒 ----
    if (!profile) {
        auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) {
            sgemm_tile16<<<grid16, block16>>>(dA, dB, dC, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            ++spins;
        }
        CUDA_CHECK(cudaGetLastError());
        printf("GPU 唤醒：先跑了 %d 次 tile16（约 0.5 秒），不计时\n\n", spins);
    }

    printf("%-16s %10s %10s %10s %12s %s\n", "版本", "平均 ms", "最快 ms", "GFLOPS", "max_rel_err",
           "结果");

    // multiple：这个版本要求 N 是几的倍数（1 表示不要求）
    auto run = [&](const char* name, int multiple, auto launch) {
        if (!wanted(only, name)) return;
#ifndef NO_N_CHECK
        if (n % multiple != 0) {
            printf("%-16s 跳过（N 必须是 %d 的倍数）\n", name, multiple);
            return;
        }
#endif
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

    run("tile16", 1, [&] { sgemm_tile16<<<grid16, block16>>>(dA, dB, dC, n); });
    run("1d_tm8", 64, [&] { sgemm_1d_tm8<<<grid1d, block1d>>>(dA, dB, dC, n); });
    run("2d_8x8", 128, [&] { sgemm_2d_8x8<<<grid2d, block2d>>>(dA, dB, dC, n); });
    run("2d_vec", 128, [&] { sgemm_2d_vec<<<grid2d, block2d>>>(dA, dB, dC, n); });
#ifdef USE_CUBLAS
    run("cublas", 1, [&] {
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
