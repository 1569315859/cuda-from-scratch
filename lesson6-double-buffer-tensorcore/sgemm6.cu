// sgemm6.cu —— 第 6 课：双缓冲、异步拷贝（cp.async）、warp 分块、Tensor Core（WMMA）—— GEMM 收尾
// C = A * B，行优先（row-major），N x N。FP32 版本输入输出都是 float；
// WMMA 版本输入是 half（FP16，由 float 转换而来），累加和输出是 float。
//
// 版本（每一步只加一样新东西）：
//   2d_vec       第 5 课的 sgemm_2d_vec（原样复制），当「基线」
//   db_reg       + 双缓冲（寄存器预取）：算当前块的同时，把下一块先读进寄存器；两份共享内存，每轮只同步 1 次
//   db_cpasync   + 异步拷贝 cp.async（sm_80 起才有）：全局 → 共享，不经过寄存器，两级流水线
//   warptile     + warp 分块：block 128x128 → warp 64x32 → 线程 8x8（拆成 2x2 个 4x4），共享读更集中、bank 更友好
//   wmma_naive   Tensor Core 入门：每个 warp 用 WMMA 16x16x16 直接从全局内存算一个 16x16 的 C（不分块）
//   wmma_fp16    Tensor Core + 前面所有技巧：128x128 block、64x32 warp 分块、cp.async 双缓冲
//   cublas       cublasSgemm（FP32，CUDA core）
//   cublas_tf32  cublasGemmEx，float 输入，TF32 Tensor Core 计算
//   cublas_fp16  cublasGemmEx，half 输入，FP32 累加（Tensor Core）
//
// 用法：
//   ./sgemm6 [N] [--profile] [--only=2d_vec,db_reg,db_cpasync,warptile,wmma_naive,wmma_fp16,cublas,cublas_tf32,cublas_fp16]
//   N 默认 1024。除 wmma_naive（要求 32 的倍数）外，所有自己写的版本要求 N 是 128 的倍数，否则自动跳过。
//   --profile：每个版本只跑 1 次、不计时（给 ncu / compute-sanitizer 用）
//   正确性：FP32 版本 max_rel_err < 1e-3；TF32 / FP16 版本 < 1e-2（输入精度只有 10 位尾数，见讲义第 7 章）
//
// 故意制造 bug 的编译开关（只用于讲义第 10 章的演示）：
//   -DSKIP_CP_WAIT   去掉 cp.async 的 wait_group（db_cpasync / warptile / wmma_fp16 三个版本都受影响）
//   -DSKIP_DB_SYNC   去掉 db_reg 里唯一的那个 __syncthreads()

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>  // half、__float2half_rn
#include <mma.h>        // nvcuda::wmma（Tensor Core 的 C++ 接口）

#include "common.h"    // fill_random / matmul_ikj_serial / max_rel_err / gflops（原样沿用）
#include "common.cuh"  // CUDA_CHECK / CUBLAS_CHECK（原样沿用）

#ifndef REF_MAX_N
#define REF_MAX_N 2048  // N 不超过这个值时用 CPU 串行结果当标准答案；更大时改用 cuBLAS（FP32）
#endif

// 第 5 课基线里的两个同步宏（本课不拿它们做 bug 演示，始终是 __syncthreads()）
#define SYNC_AFTER_LOAD() __syncthreads()
#define SYNC_AFTER_COMPUTE() __syncthreads()

#ifdef SKIP_DB_SYNC
#define DB_SYNC()                  // bug 演示：db_reg 去掉唯一的同步
#else
#define DB_SYNC() __syncthreads()
#endif

// ---------------------------------------------------------------------------
// cp.async 小工具（内联 PTX）。cp.async 是 sm_80（Ampere）才有的指令，你的 RTX 2050 是 sm_86，支持。
//   cp_async_4  ：拷 4 字节（1 个 float），只能用 .ca（数据也放进 L1）
//   cp_async_16 ：拷 16 字节（4 个 float / 8 个 half），用 .cg（绕过 L1，只经过 L2）
//   cp_async_commit   ：把「到目前为止发出的、还没打包的拷贝」打成一个组（group）
//   cp_async_wait_all ：等到本线程发出的所有组都完成（wait_group 0）
// 注意：wait 只保证「本线程自己发的拷贝」完成；要看到别的线程拷进来的数据，后面还要 __syncthreads()。
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#error "本课的 cp.async / WMMA 版本需要 sm_80 及以上，请用 -arch=sm_86 编译"
#endif

__device__ __forceinline__ void cp_async_4(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));  // 共享内存的 32 位地址
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

#ifdef SKIP_CP_WAIT
#define CP_ASYNC_WAIT_ALL()        // bug 演示：不等拷贝落地就去读共享内存
#else
#define CP_ASYNC_WAIT_ALL() cp_async_wait_all()
#endif

// 尺寸参数（和第 5 课一样）
namespace cfg2d {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);   // 16*16 = 256
}  // namespace cfg2d

// warp 分块的尺寸
namespace cfgwt {
constexpr int BM = 128, BN = 128, BK = 8;
constexpr int WM = 64, WN = 32;                  // 一个 warp 负责 C 的 64 行 x 32 列
constexpr int WARPS_N = BN / WN;                 // 横向 4 个 warp
constexpr int WARPS_M = BM / WM;                 // 纵向 2 个 warp
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 8 个 warp = 256 个线程
constexpr int LANES_N = 4;                       // warp 内 32 个线程排成 8 行 x 4 列
constexpr int LANES_M = 32 / LANES_N;            // 8
constexpr int SUBM = WM / 2;                     // 32：线程的 8 行拆成 2 段，每段 4 行，两段相隔 32 行
constexpr int SUBN = WN / 2;                     // 16：线程的 8 列拆成 2 段，每段 4 列，两段相隔 16 列
constexpr int TM = 8, TN = 8;
static_assert(LANES_M * 4 * 2 == WM, "LANES_M * 4 * 2 must equal WM");
static_assert(LANES_N * 4 * 2 == WN, "LANES_N * 4 * 2 must equal WN");
}  // namespace cfgwt

// WMMA 分块的尺寸（元素类型是 half）
namespace cfgwm {
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WM = 64, WN = 32;                  // 一个 warp 负责 64 x 32 = 4 x 2 个 16x16 小块
constexpr int WARPS_N = BN / WN;                 // 4
constexpr int THREADS = (BM / WM) * WARPS_N * 32;  // 256
constexpr int FM = WM / 16, FN = WN / 16;        // 4, 2
constexpr int PAD = 8;                           // 每行多留 8 个 half（16 字节），错开 bank
constexpr int LDA = BK + PAD;                    // As 每行 40 个 half
constexpr int LDB = BN + PAD;                    // Bs 每行 136 个 half
}  // namespace cfgwm

// ======== GPU kernels 开始 ========

// ---------------------------------------------------------------------------
// 基线：第 5 课的 sgemm_2d_vec（原样复制）
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

// ---------------------------------------------------------------------------
// 小工具：一个线程在一个 dot 上做 8x8 外积（三个 FP32 kernel 共用，和 2d_vec 的内层完全一样）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void outer_8x8(float* acc, const float4 m0, const float4 m1,
                                          const float4 n0, const float4 n1) {
    const float regM[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    const float regN[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i * 8 + j] += regM[i] * regN[j];
}

// 小工具：把一个 float4（A 的同一行、连续 4 个 k）转置写进 As[k][m]（和 2d_vec 一样）
__device__ __forceinline__ void store_a_transposed(float* As, const float4 a4, int innerRowA,
                                                   int innerColA) {
    constexpr int BM = cfg2d::BM;
    As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
}

// ---------------------------------------------------------------------------
// v1：双缓冲（寄存器预取）sgemm_db_reg
//   分工、计算和 2d_vec 一模一样，只改「什么时候搬」：
//   ① 循环开始先发出「下一块」的全局读（结果进寄存器 a4 / b4，不等它回来）
//   ② 用当前那份共享内存 buf[cur] 做 8 个 dot 的外积 —— 这段时间正好盖住全局读的延迟
//   ③ 算完再把寄存器里的下一块写进另一份共享内存 buf[cur^1]
//   ④ 一次 __syncthreads()，交换 cur
//   共享内存 2 份 x 8 KB = 16 KB。
// ---------------------------------------------------------------------------
// 注意 __launch_bounds__ 的第二个参数 2：「每个 SM 至少放 2 个 block」→ 每线程最多 128 个寄存器。
// 不加它时编译器给了 129 个（多出来的是预取用的 a4 / b4），取整到 136 后每 SM 只能放 1 个 block（见讲义 3.6）。
__global__ void __launch_bounds__(cfg2d::THREADS, 2)
sgemm_db_reg(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int n) {
    using namespace cfg2d;
    __shared__ __align__(16) float As[2][BK * BM];   // 两份，每份是转置后的 8 x 128
    __shared__ __align__(16) float Bs[2][BK * BN];   // 两份，每份 8 x 128

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN);  // 0..15
    const int threadRow = tid / (BN / TN);  // 0..15
    const int innerRowA = tid / (BK / 4);   // 0..127
    const int innerColA = tid % (BK / 4);   // 0..1
    const int innerRowB = tid / (BN / 4);   // 0..7
    const int innerColB = tid % (BN / 4);   // 0..31

    const float* Ablk = A + (size_t)cRow * BM * n;
    const float* Bblk = B + (size_t)cCol * BN;

    float acc[TM * TN] = {0.0f};

    // ---- 序幕：第 0 块 全局 → 寄存器 → buf 0 ----
    float4 a4 = *reinterpret_cast<const float4*>(&Ablk[(size_t)innerRowA * n + innerColA * 4]);
    float4 b4 = *reinterpret_cast<const float4*>(&Bblk[(size_t)innerRowB * n + innerColB * 4]);
    store_a_transposed(As[0], a4, innerRowA, innerColA);
    *reinterpret_cast<float4*>(&Bs[0][innerRowB * BN + innerColB * 4]) = b4;
    __syncthreads();

    int cur = 0;
    for (int bk = 0; bk < n; bk += BK) {
        const bool hasNext = (bk + BK < n);
        // ① 发出下一块的全局读：只是「发出去」，数据回来之前线程可以继续往下执行
        if (hasNext) {
            a4 = *reinterpret_cast<const float4*>(
                &Ablk[(size_t)innerRowA * n + (bk + BK) + innerColA * 4]);
            b4 = *reinterpret_cast<const float4*>(
                &Bblk[(size_t)(bk + BK + innerRowB) * n + innerColB * 4]);
        }
        // ② 用 buf[cur] 计算（和 2d_vec 的内层一样）
#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            const float4* aRow = reinterpret_cast<const float4*>(&As[cur][dot * BM + threadRow * TM]);
            const float4* bRow = reinterpret_cast<const float4*>(&Bs[cur][dot * BN + threadCol * TN]);
            outer_8x8(acc, aRow[0], aRow[1], bRow[0], bRow[1]);
        }
        // ③ 把寄存器里的下一块写进另一份共享内存（这时 a4 / b4 一定已经回来了）
        if (hasNext) {
            store_a_transposed(As[cur ^ 1], a4, innerRowA, innerColA);
            *reinterpret_cast<float4*>(&Bs[cur ^ 1][innerRowB * BN + innerColB * 4]) = b4;
        }
        // ④ 唯一的同步：保证 (a) 大家都写完了 buf[cur^1]，下一轮才能读；
        //                    (b) 大家都读完了 buf[cur]，下下轮才能往里写
        DB_SYNC();
        cur ^= 1;
    }

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

// ---------------------------------------------------------------------------
// 小工具：用 cp.async 把第 bk 块搬进一份共享内存（db_cpasync 和 warptile 共用）
//   A：每个线程负责 A 小块同一行的 4 个连续 float，用 4 条 4 字节 cp.async 直接送到转置后的位置
//      （cp.async 是「原样搬一段字节」，不能在路上转置 16 字节的整块，所以拆成 4 字节）
//   B：每个线程 1 条 16 字节 cp.async，原样搬
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_tile_async(float* As, float* Bs, const float* Ablk,
                                                const float* Bblk, int n, int bk, int innerRowA,
                                                int innerColA, int innerRowB, int innerColB) {
    constexpr int BM = cfg2d::BM, BN = cfg2d::BN;
    const float* aSrc = &Ablk[(size_t)innerRowA * n + bk + innerColA * 4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        cp_async_4(&As[(innerColA * 4 + q) * BM + innerRowA], aSrc + q);
    cp_async_16(&Bs[innerRowB * BN + innerColB * 4],
                &Bblk[(size_t)(bk + innerRowB) * n + innerColB * 4]);
}

// ---------------------------------------------------------------------------
// v2：cp.async 两级流水线 sgemm_db_cpasync
//   每轮的顺序：等第 t 块落地 → 同步 → 发出第 t+1 块的异步拷贝 → 算第 t 块
//   和 db_reg 比：数据不再经过寄存器（省掉 a4/b4 这 8 个寄存器和 STS 指令），
//   拷贝由 LSU 后台完成，线程发完指令马上去算。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(cfg2d::THREADS)
sgemm_db_cpasync(const float* __restrict__ A, const float* __restrict__ B,
                 float* __restrict__ C, int n) {
    using namespace cfg2d;
    __shared__ __align__(16) float As[2][BK * BM];
    __shared__ __align__(16) float Bs[2][BK * BN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const int innerRowA = tid / (BK / 4);
    const int innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BN / 4);
    const int innerColB = tid % (BN / 4);

    const float* Ablk = A + (size_t)cRow * BM * n;
    const float* Bblk = B + (size_t)cCol * BN;

    float acc[TM * TN] = {0.0f};

    // 序幕：发出第 0 块的拷贝，打包成第 0 组
    load_tile_async(As[0], Bs[0], Ablk, Bblk, n, 0, innerRowA, innerColA, innerRowB, innerColB);
    cp_async_commit();

    int cur = 0;
    for (int bk = 0; bk < n; bk += BK) {
        CP_ASYNC_WAIT_ALL();   // ① 等「我自己发的」第 t 块拷贝落地
        __syncthreads();       // ② 等「所有人的」第 t 块落地；也保证大家都算完了第 t-1 块（buf[cur^1] 空了）
        if (bk + BK < n) {     // ③ 发出第 t+1 块 → buf[cur^1]，不等它
            load_tile_async(As[cur ^ 1], Bs[cur ^ 1], Ablk, Bblk, n, bk + BK, innerRowA,
                            innerColA, innerRowB, innerColB);
            cp_async_commit();
        }
        // ④ 算第 t 块（这段时间里第 t+1 块正在后台搬）
#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            const float4* aRow = reinterpret_cast<const float4*>(&As[cur][dot * BM + threadRow * TM]);
            const float4* bRow = reinterpret_cast<const float4*>(&Bs[cur][dot * BN + threadCol * TN]);
            outer_8x8(acc, aRow[0], aRow[1], bRow[0], bRow[1]);
        }
        cur ^= 1;
    }

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

// ---------------------------------------------------------------------------
// v3：warp 分块 sgemm_warptile（在 db_cpasync 基础上只改「线程 ↔ C 的对应关系」）
//   三层：block 负责 128x128 → 8 个 warp 排成 2 x 4，每个 warp 负责 64x32 → warp 里 32 个线程排成 8 x 4，
//   每个线程负责 8x8 个 C，但不是连成一块，而是 2x2 个 4x4 的小块：
//     行：warp 起点 + mi*32 + laneRow*4 + 0..3（mi = 0, 1）
//     列：warp 起点 + ni*16 + laneCol*4 + 0..3（ni = 0, 1）
//   好处：一条 LDS.128 里，一个 warp 读的共享内存地址挤在一段连续的 128 字节（A）/ 64 字节（B）里，
//   1 个 wavefront 就够；2d_vec 读 B 时 16 个线程的地址散在 512 字节里，要多个 wavefront。
//   每线程仍是 8+8 个数换 64 次 FMA，寄存器复用不变。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(cfgwt::THREADS)
sgemm_warptile(const float* __restrict__ A, const float* __restrict__ B,
               float* __restrict__ C, int n) {
    using namespace cfgwt;
    __shared__ __align__(16) float As[2][BK * BM];
    __shared__ __align__(16) float Bs[2][BK * BN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.x;

    // 计算阶段的分工：block → warp → 线程
    const int warpId = tid / 32;            // 0..7
    const int lane = tid % 32;              // 0..31
    const int warpRow = warpId / WARPS_N;   // 0..1
    const int warpCol = warpId % WARPS_N;   // 0..3
    const int laneRow = lane / LANES_N;     // 0..7
    const int laneCol = lane % LANES_N;     // 0..3
    const int mBase = warpRow * WM + laneRow * 4;   // 本线程第一段 4 行在 C 小块里的起点
    const int nBase = warpCol * WN + laneCol * 4;   // 本线程第一段 4 列在 C 小块里的起点

    // 搬运阶段的分工：和 2d_vec / db_cpasync 完全一样
    const int innerRowA = tid / (BK / 4);
    const int innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BN / 4);
    const int innerColB = tid % (BN / 4);

    const float* Ablk = A + (size_t)cRow * BM * n;
    const float* Bblk = B + (size_t)cCol * BN;

    float acc[TM * TN] = {0.0f};   // acc[i*8 + j]：i = mi*4 + r，j = ni*4 + c

    load_tile_async(As[0], Bs[0], Ablk, Bblk, n, 0, innerRowA, innerColA, innerRowB, innerColB);
    cp_async_commit();

    int cur = 0;
    for (int bk = 0; bk < n; bk += BK) {
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
        if (bk + BK < n) {
            load_tile_async(As[cur ^ 1], Bs[cur ^ 1], Ablk, Bblk, n, bk + BK, innerRowA,
                            innerColA, innerRowB, innerColB);
            cp_async_commit();
        }
#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            const float* aK = &As[cur][dot * BM];   // As 第 dot 行（128 个 m）
            const float* bK = &Bs[cur][dot * BN];   // Bs 第 dot 行（128 个 n）
            const float4 m0 = *reinterpret_cast<const float4*>(&aK[mBase]);          // 行段 0
            const float4 m1 = *reinterpret_cast<const float4*>(&aK[mBase + SUBM]);   // 行段 1（+32 行）
            const float4 n0 = *reinterpret_cast<const float4*>(&bK[nBase]);          // 列段 0
            const float4 n1 = *reinterpret_cast<const float4*>(&bK[nBase + SUBN]);   // 列段 1（+16 列）
            outer_8x8(acc, m0, m1, n0, n1);
        }
        cur ^= 1;
    }

    // 写 C：8 行，每行 2 段 x 4 个连续列 → 每行 2 条 float4 写
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = cRow * BM + mBase + (i / 4) * SUBM + (i % 4);
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int col = cCol * BN + nBase + ni * SUBN;
            *reinterpret_cast<float4*>(&C[(size_t)row * n + col]) =
                make_float4(acc[i * TN + ni * 4 + 0], acc[i * TN + ni * 4 + 1],
                            acc[i * TN + ni * 4 + 2], acc[i * TN + ni * 4 + 3]);
        }
    }
}

// ---------------------------------------------------------------------------
// float → half 转换（只在计时之前做一次，不计入 WMMA 版本的时间）
// ---------------------------------------------------------------------------
__global__ void convert_f32_to_f16(const float* __restrict__ in, half* __restrict__ out, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half_rn(in[i]);   // rn = 四舍五入到最近（round to nearest）
}

// ---------------------------------------------------------------------------
// v4a：Tensor Core 入门 sgemm_wmma_naive
//   一个 block 4 个 warp（128 线程），负责 C 的 32x32；每个 warp 负责一个 16x16。
//   沿 k 每次走 16：从全局内存直接装一个 A 的 16x16、一个 B 的 16x16，做一次 mma_sync。
//   没有共享内存、没有分块复用 —— 目的是看清 WMMA 的 3 个步骤：装（load）→ 乘加（mma）→ 存（store）。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
sgemm_wmma_naive(const half* __restrict__ A, const half* __restrict__ B,
                 float* __restrict__ C, int n) {
    using namespace nvcuda;
    const int warpId = threadIdx.x / 32;           // 0..3
    const int row0 = blockIdx.y * 32 + (warpId / 2) * 16;
    const int col0 = blockIdx.x * 32 + (warpId % 2) * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int k = 0; k < n; k += 16) {
        wmma::load_matrix_sync(aFrag, A + (size_t)row0 * n + k, n);   // A[row0..+15][k..k+15]
        wmma::load_matrix_sync(bFrag, B + (size_t)k * n + col0, n);   // B[k..k+15][col0..+15]
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);                    // cFrag += aFrag * bFrag
    }
    wmma::store_matrix_sync(C + (size_t)row0 * n + col0, cFrag, n, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// v4b：Tensor Core + 分块 + cp.async 双缓冲 sgemm_wmma_fp16
//   block 128x128，8 个 warp 排成 2 x 4，每个 warp 负责 64x32 = 4 x 2 个 16x16（8 个累加器 fragment）。
//   每轮沿 k 搬 32：A 小块 128x32 half、B 小块 32x128 half，用 16 字节 cp.async（每次 8 个 half）。
//   共享内存每行多留 PAD = 8 个 half，减少 WMMA 从共享内存装 fragment 时的 bank 冲突。
//   共享内存：2 x (128x40 + 32x136) x 2 字节 = 37,888 字节 < 48 KB，静态声明即可。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(cfgwm::THREADS)
sgemm_wmma_fp16(const half* __restrict__ A, const half* __restrict__ B,
                float* __restrict__ C, int n) {
    using namespace nvcuda;
    using namespace cfgwm;
    __shared__ __align__(128) half As[2][BM * LDA];   // 2 份，每份 128 行 x 40（32 有效 + 8 填充）
    __shared__ __align__(128) half Bs[2][BK * LDB];   // 2 份，每份 32 行 x 136（128 有效 + 8 填充）

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / WARPS_N;   // 0..1
    const int warpCol = warpId % WARPS_N;   // 0..3

    const half* Ablk = A + (size_t)cRow * BM * n;
    const half* Bblk = B + (size_t)cCol * BN;

    // 搬运：A 小块 128 行 x 32 个 half = 每行 4 个 16 字节块，共 512 块；B 小块 32 行 x 128 = 每行 16 块，共 512 块
    // 256 个线程，每人 A 2 块、B 2 块
    auto load_tile = [&](int buf, int bk) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int c = tid + r * THREADS;          // 0..511
            const int aRow = c / (BK / 8), aCol = (c % (BK / 8)) * 8;
            cp_async_16(&As[buf][aRow * LDA + aCol], &Ablk[(size_t)aRow * n + bk + aCol]);
            const int bRow = c / (BN / 8), bCol = (c % (BN / 8)) * 8;
            cp_async_16(&Bs[buf][bRow * LDB + bCol], &Bblk[(size_t)(bk + bRow) * n + bCol]);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag[FM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag[FN];

    load_tile(0, 0);
    cp_async_commit();

    int cur = 0;
    for (int bk = 0; bk < n; bk += BK) {
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
        if (bk + BK < n) {
            load_tile(cur ^ 1, bk + BK);
            cp_async_commit();
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {        // 32 = 2 步 x 16
#pragma unroll
            for (int i = 0; i < FM; ++i)
                wmma::load_matrix_sync(aFrag[i], &As[cur][(warpRow * WM + i * 16) * LDA + kk], LDA);
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::load_matrix_sync(bFrag[j], &Bs[cur][kk * LDB + warpCol * WN + j * 16], LDB);
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        cur ^= 1;
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            float* cPtr = C + (size_t)(cRow * BM + warpRow * WM + i * 16) * n
                            + cCol * BN + warpCol * WN + j * 16;
            wmma::store_matrix_sync(cPtr, acc[i][j], n, wmma::mem_row_major);
        }
}

// ======== GPU kernels 结束 ========

// ---------------------------------------------------------------------------
// 计时：warmup 次预热（不计时）+ reps 次计时（和第 4、5 课完全一样）
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

// 打印一个 kernel 的「资源账单」：每线程寄存器、静态共享内存、local 溢出，
// 以及 runtime 按这些资源算出来的每 SM 最多 block 数、理论占用率
template <typename Kernel>
static void print_resources(const char* name, Kernel kernel, int threadsPerBlock,
                            const cudaDeviceProp& prop, size_t dynSmem = 0) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, threadsPerBlock,
                                                             dynSmem));
    const double occ = 100.0 * blocksPerSM * threadsPerBlock / prop.maxThreadsPerMultiProcessor;
    printf("%-11s %8d %10d %12zu %14zu %12d %10.1f%%\n", name, threadsPerBlock, attr.numRegs,
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
    if (prop.major < 8) {
        fprintf(stderr, "这台 GPU 是 sm_%d%d，cp.async / 本课的 WMMA 写法需要 sm_80 及以上\n", prop.major,
                prop.minor);
        return 1;
    }
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const size_t nn = (size_t)n * n;
    const size_t bytes = nn * sizeof(float);
    const size_t bytesH = nn * sizeof(half);
    const size_t need = 3 * bytes + 2 * bytesH;   // A、B、C（float）+ A、B 的 half 副本
    printf("GPU: %s (sm_%d%d), SM 数 %d, 每 SM 寄存器 %d 个, 每 SM 最多 %d 线程, 每 SM 共享内存 %zu KB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.regsPerMultiprocessor,
           prop.maxThreadsPerMultiProcessor, prop.sharedMemPerMultiprocessor / 1024);
    printf("显存: 空闲 %.0f MB / 总共 %.0f MB；本次需要 %.1f MB\n", freeB / 1048576.0,
           totalB / 1048576.0, need / 1048576.0);
    if (need > freeB * 0.8) {
        fprintf(stderr, "N=%d 需要的显存太多，请减小 N\n", n);
        return 1;
    }
    printf("N = %d, 模式 = %s\n\n", n, profile ? "profile（每个版本只跑 1 次，不计时）"
                                              : "benchmark（GPU 唤醒 0.5 秒 + 每版 1 次预热 + 10 次计时）");

    // ---- 2. 资源账单（编译器实际分配的寄存器数，运行时查询）----
    // 本课所有 kernel 的共享内存都 ≤ 48 KB，用静态声明即可，不需要 cudaFuncSetAttribute 申请（见讲义坑 6、练习 W3）
    if (!profile) {
        printf("%-11s %8s %10s %12s %14s %12s %11s\n", "kernel", "线程/blk", "寄存器/线程",
               "静态smem(B)", "local溢出(B)", "blocks/SM", "理论占用率");
        print_resources("2d_vec", sgemm_2d_vec, cfg2d::THREADS, prop);
        print_resources("db_reg", sgemm_db_reg, cfg2d::THREADS, prop);
        print_resources("db_cpasync", sgemm_db_cpasync, cfg2d::THREADS, prop);
        print_resources("warptile", sgemm_warptile, cfgwt::THREADS, prop);
        print_resources("wmma_naive", sgemm_wmma_naive, 128, prop);
        print_resources("wmma_fp16", sgemm_wmma_fp16, cfgwm::THREADS, prop);
        printf("\n");
    }

    // ---- 3. 主机端数据 ----
    std::vector<float> hA(nn), hB(nn), hC(nn), hRef(nn);
    fill_random(hA, 1);
    fill_random(hB, 2);

    // ---- 4. 设备端数据 ----
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    half *dA16 = nullptr, *dB16 = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMalloc(&dA16, bytesH));
    CUDA_CHECK(cudaMalloc(&dB16, bytesH));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    {   // float → half，只做一次，不计时
        const int t = 256;
        const unsigned g = (unsigned)((nn + t - 1) / t);
        convert_f32_to_f16<<<g, t>>>(dA, dA16, nn);
        convert_f32_to_f16<<<g, t>>>(dB, dB16, nn);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

#ifdef USE_CUBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
#endif

    // ---- 5. 标准答案（FP32）----
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
        refName = "cuBLAS FP32";
    }
#endif
    printf("正确性标准答案：%s（FP32 版本要求 < 1e-3，TF32 / FP16 版本要求 < 1e-2）\n\n", refName);

    // ---- 6. 启动配置 ----
    const int warmup = profile ? 0 : 1;
    const int reps = profile ? 1 : 10;
    dim3 block2d(cfg2d::THREADS);                         // 256 个线程，一维
    dim3 grid128(n / 128, n / 128);                       // 所有 128x128 版本共用（N 是 128 的倍数时才会用到）
    dim3 blockWmmaNaive(128);                             // 4 个 warp
    dim3 gridWmmaNaive(n / 32, n / 32);                   // 每个 block 负责 32x32

    // ---- 唤醒 GPU：先连续跑 2d_vec 至少 0.5 秒（第 4 课的教训）----
    if (!profile && n % 128 == 0) {
        auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) {
            sgemm_2d_vec<<<grid128, block2d>>>(dA, dB, dC, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            ++spins;
        }
        CUDA_CHECK(cudaGetLastError());
        printf("GPU 唤醒：先跑了 %d 次 2d_vec（约 0.5 秒），不计时\n\n", spins);
    }

    printf("%-16s %10s %10s %10s %12s %s\n", "版本", "平均 ms", "最快 ms", "GFLOPS", "max_rel_err",
           "结果");

    // multiple：这个版本要求 N 是几的倍数；tol：误差上限
    auto run = [&](const char* name, int multiple, double tol, auto launch) {
        if (!wanted(only, name)) return;
        if (n % multiple != 0) {
            printf("%-16s 跳过（N 必须是 %d 的倍数）\n", name, multiple);
            return;
        }
        CUDA_CHECK(cudaMemset(dC, 0, bytes));  // 清零，防止上一个版本的结果「冒充」本版本
        Timing t = time_gpu(launch, warmup, reps);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double err = haveRef ? max_rel_err(hC.data(), hRef.data(), nn) : -1.0;
        const char* verdict = !haveRef ? "未检查" : (err < tol ? "OK" : "WRONG");
        if (profile)
            printf("%-16s %10s %10s %10s %12.2e %s\n", name, "-", "-", "-", err, verdict);
        else
            printf("%-16s %10.3f %10.3f %10.1f %12.2e %s\n", name, t.avg_ms, t.min_ms,
                   gflops(n, t.avg_ms / 1e3), err, verdict);
    };

    run("2d_vec", 128, 1e-3, [&] { sgemm_2d_vec<<<grid128, block2d>>>(dA, dB, dC, n); });
    run("db_reg", 128, 1e-3, [&] { sgemm_db_reg<<<grid128, block2d>>>(dA, dB, dC, n); });
    run("db_cpasync", 128, 1e-3, [&] { sgemm_db_cpasync<<<grid128, block2d>>>(dA, dB, dC, n); });
    run("warptile", 128, 1e-3,
        [&] { sgemm_warptile<<<grid128, dim3(cfgwt::THREADS)>>>(dA, dB, dC, n); });
    run("wmma_naive", 32, 1e-2,
        [&] { sgemm_wmma_naive<<<gridWmmaNaive, blockWmmaNaive>>>(dA16, dB16, dC, n); });
    run("wmma_fp16", 128, 1e-2,
        [&] { sgemm_wmma_fp16<<<grid128, dim3(cfgwm::THREADS)>>>(dA16, dB16, dC, n); });
#ifdef USE_CUBLAS
    run("cublas", 1, 1e-3, [&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n,
                                 &beta, dC, n));
    });
    run("cublas_tf32", 1, 1e-2, [&] {   // float 输入，计算时把尾数截到 TF32（10 位），走 Tensor Core
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, CUDA_R_32F, n,
                                  dA, CUDA_R_32F, n, &beta, dC, CUDA_R_32F, n,
                                  CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    });
    run("cublas_fp16", 1, 1e-2, [&] {   // half 输入，FP32 累加，float 输出，走 Tensor Core
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB16, CUDA_R_16F,
                                  n, dA16, CUDA_R_16F, n, &beta, dC, CUDA_R_32F, n,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    });
    CUBLAS_CHECK(cublasDestroy(handle));
#endif

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dA16));
    CUDA_CHECK(cudaFree(dB16));
    printf("\nGFLOPS 按平均时间计算 = 2*N^3 / 平均时间（FP16 / TF32 版本也按同一公式，便于横向比较）。\n");
    return 0;
}
