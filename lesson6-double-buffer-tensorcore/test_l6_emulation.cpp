// test_l6_emulation.cpp —— 在没有 GPU 的机器上检查 sgemm6.cu 里 6 个 kernel 的下标数学和流水线顺序
//
// 做法（和第 5 课的 test_index_emulation.cpp 同一个思路）：
//   把 sgemm6.cu 里从「尺寸参数」到「GPU kernels 结束」之间的源码原样抽出来，用一组 shim（替身）在 CPU 上编译：
//   - 一个 block 的每个 CUDA 线程对应一个 std::thread，__syncthreads 用 std::barrier 实现，block 依次执行；
//   - cp.async 替身按「最坏情况」模拟异步：cp_async_4/16 只是记下来，commit 把它们打成一组，
//     只有 wait（cp_async_wait_all）时才真正拷贝。所以少了 wait，数据就永远不会落地 —— 用来演示 -DSKIP_CP_WAIT；
//   - WMMA 替身：每个线程手里都有一份完整的 16x16 fragment（真硬件是 32 个线程分着拿），
//     load / mma / store 按定义逐元素算，store 只让 lane 0 写；同时检查 WMMA 的对齐要求
//     （指针 32 字节对齐、ldm 是 16 字节的倍数）；
//   - 矩阵放在 mmap 出来的页里，前后各留一页 PROT_NONE 保护页，越界读写直接段错误。
//   每个 kernel 都和 CPU 参考矩阵乘比较；C 先填 NaN，漏写的格子一定会被发现。
//
// 构建与运行（需要 C++20 的 <barrier> 和 _Float16；g++ 12+ 或 clang++ 15+）：
//   sed -n '/尺寸参数（和第 5 课一样）/,/GPU kernels 结束/p' sgemm6.cu > kernels6_extracted.inc
//   g++ -std=c++20 -O2 -pthread test_l6_emulation.cpp -o test_l6_emulation
//   ./test_l6_emulation
//   加 -DSKIP_CP_WAIT 编译：三个 cp.async 版本应该 WRONG；加 -DSKIP_DB_SYNC：db_reg 可能 WRONG（不保证每次）。

#include <barrier>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHK(cond) do { if (!(cond)) { fprintf(stderr, "断言失败 %s 行 %d: %s\n", __FILE__, __LINE__, #cond); std::exit(2); } } while (0)

// ---------------------------- CUDA shim ----------------------------
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __restrict__ __restrict

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
thread_local dim3 threadIdx, blockIdx, blockDim;
thread_local std::barrier<>* g_bar = nullptr;
inline void __syncthreads() { g_bar->arrive_and_wait(); }

struct __attribute__((aligned(16))) float4 { float x, y, z, w; };
inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }

using half = _Float16;
inline half __float2half_rn(float f) { return (half)f; }

// cp.async 替身：记账，wait 时才拷
struct PendingCopy { void* dst; const void* src; int bytes; };
thread_local std::vector<PendingCopy> t_uncommitted;
thread_local std::vector<std::vector<PendingCopy>> t_groups;
inline void cp_async_4(void* s, const void* g) {
    CHK((uintptr_t)s % 4 == 0 && (uintptr_t)g % 4 == 0);
    t_uncommitted.push_back({s, g, 4});
}
inline void cp_async_16(void* s, const void* g) {
    CHK((uintptr_t)s % 16 == 0 && (uintptr_t)g % 16 == 0);   // 16 字节 cp.async 两边都必须 16 字节对齐
    t_uncommitted.push_back({s, g, 16});
}
inline void cp_async_commit() { t_groups.push_back(t_uncommitted); t_uncommitted.clear(); }
inline void cp_async_wait_all() {
    for (auto& grp : t_groups)
        for (auto& c : grp) std::memcpy(c.dst, c.src, c.bytes);
    t_groups.clear();
}
#ifdef SKIP_CP_WAIT
#define CP_ASYNC_WAIT_ALL()
#else
#define CP_ASYNC_WAIT_ALL() cp_async_wait_all()
#endif
#define SYNC_AFTER_LOAD() __syncthreads()
#define SYNC_AFTER_COMPUTE() __syncthreads()
#ifdef SKIP_DB_SYNC
#define DB_SYNC()
#else
#define DB_SYNC() __syncthreads()
#endif

// WMMA 替身
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {};
struct row_major {}; struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <class Use, int M, int N, int K, class T, class Layout = void>
struct fragment { T x[16 * 16]; static constexpr int num_elements = 256; };
template <class Use, class T, class L>
inline void load_matrix_sync(fragment<Use, 16, 16, 16, T, L>& f, const T* p, unsigned ldm) {
    CHK((uintptr_t)p % 32 == 0);                 // WMMA 要求 256 位（32 字节）对齐
    CHK((ldm * sizeof(T)) % 16 == 0);            // ldm 必须是 16 字节的倍数
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) f.x[r * 16 + c] = p[(size_t)r * ldm + c];
}
template <class T>
inline void fill_fragment(fragment<accumulator, 16, 16, 16, T>& f, T v) {
    for (auto& e : f.x) e = v;
}
inline void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                     const fragment<matrix_a, 16, 16, 16, half, row_major>& a,
                     const fragment<matrix_b, 16, 16, 16, half, row_major>& b,
                     const fragment<accumulator, 16, 16, 16, float>& c) {
    float tmp[256];
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float s = c.x[i * 16 + j];
            for (int k = 0; k < 16; ++k) s += (float)a.x[i * 16 + k] * (float)b.x[k * 16 + j];
            tmp[i * 16 + j] = s;
        }
    std::memcpy(d.x, tmp, sizeof(tmp));
}
inline void store_matrix_sync(float* p, const fragment<accumulator, 16, 16, 16, float>& f,
                              unsigned ldm, layout_t) {
    CHK((uintptr_t)p % 32 == 0);
    CHK((ldm * sizeof(float)) % 16 == 0);
    if (threadIdx.x % 32 != 0) return;           // 32 个替身线程算的一样，只让 lane 0 写
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) p[(size_t)r * ldm + c] = f.x[r * 16 + c];
}
}}  // namespace nvcuda::wmma

#include "kernels6_extracted.inc"   // sgemm6.cu 里的尺寸参数 + kernel 源码，原样

// ---------------------------- 带保护页的数组 ----------------------------
template <class T>
struct Guarded {
    void* base = nullptr;
    size_t total = 0;
    T* p = nullptr;
    explicit Guarded(size_t count) {
        const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
        const size_t bytes = count * sizeof(T);
        const size_t body = (bytes + pg - 1) / pg * pg;
        total = body + 2 * pg;
        base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHK(base != MAP_FAILED);
        mprotect(base, pg, PROT_NONE);
        mprotect((char*)base + pg + body, pg, PROT_NONE);
        p = (T*)((char*)base + pg + body - bytes);   // 数组末尾紧贴后保护页
        CHK(((uintptr_t)p % 256) == 0);              // 模仿 cudaMalloc 的 256 字节对齐（N 是 128 的倍数时成立）
    }
    ~Guarded() { munmap(base, total); }
};

template <class TIn>
static void launch(void (*k)(const TIn*, const TIn*, float*, int), dim3 grid, dim3 block,
                   const TIn* A, const TIn* B, float* C, int n) {
    const unsigned T = block.x;
    std::barrier<> bar((std::ptrdiff_t)T);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < T; ++t) {
        pool.emplace_back([&, t] {
            g_bar = &bar;
            threadIdx = dim3(t, 0, 0);
            blockDim = block;
            for (unsigned by = 0; by < grid.y; ++by)
                for (unsigned bx = 0; bx < grid.x; ++bx) {
                    blockIdx = dim3(bx, by, 0);
                    t_uncommitted.clear();
                    t_groups.clear();
                    k(A, B, C, n);
                    bar.arrive_and_wait();
                }
        });
    }
    for (auto& th : pool) th.join();
}

static void ref_matmul(const float* A, const float* B, double* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < n; ++k) s += (double)A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

static double max_err(const float* x, const std::vector<double>& ref) {
    double w = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(x[i])) return INFINITY;   // 漏写的格子
        w = std::max(w, std::fabs(x[i] - ref[i]) / (std::fabs(ref[i]) + 1.0));
    }
    return w;
}

int main() {
    int fails = 0;
    for (int n : {128, 256}) {
        const size_t nn = (size_t)n * n;
        Guarded<float> A(nn), B(nn), C(nn);
        Guarded<half> A16(nn), B16(nn);
        std::mt19937 g1(1), g2(2);
        std::uniform_real_distribution<float> d(0.f, 1.f);
        for (size_t i = 0; i < nn; ++i) A.p[i] = d(g1);
        for (size_t i = 0; i < nn; ++i) B.p[i] = d(g2);
        // half 副本：用 sgemm6.cu 里的 convert_f32_to_f16 kernel 本身来转（顺便测它）
        {
            const unsigned t = 256, g = (unsigned)((nn + t - 1) / t);
            for (int which = 0; which < 2; ++which) {
                const float* in = which ? B.p : A.p;
                half* out = which ? B16.p : A16.p;
                for (unsigned bx = 0; bx < g; ++bx) {
                    blockIdx = dim3(bx); blockDim = dim3(t);
                    for (unsigned tx = 0; tx < t; ++tx) { threadIdx = dim3(tx); convert_f32_to_f16(in, out, nn); }
                }
            }
        }
        std::vector<double> ref(nn), ref16(nn);
        ref_matmul(A.p, B.p, ref.data(), n);
        {   // 「输入先变成 half」的参考答案：用来严格检查 WMMA 版本的下标（误差只剩累加顺序）
            std::vector<float> a(nn), b(nn);
            for (size_t i = 0; i < nn; ++i) { a[i] = (float)A16.p[i]; b[i] = (float)B16.p[i]; }
            ref_matmul(a.data(), b.data(), ref16.data(), n);
        }

        auto check = [&](const char* name, double tol, double e, const char* vs) {
            const bool ok = e < tol;
            printf("N=%-4d %-12s max_rel_err=%.2e（对%s，要求 < %.0e） %s\n", n, name, e, vs, tol,
                   ok ? "OK" : "WRONG");
            if (!ok) ++fails;
        };
        auto run32 = [&](const char* name, void (*k)(const float*, const float*, float*, int), unsigned threads) {
            for (size_t i = 0; i < nn; ++i) C.p[i] = NAN;
            launch<float>(k, dim3(n / 128, n / 128), dim3(threads), A.p, B.p, C.p, n);
            check(name, 1e-5, max_err(C.p, ref), " FP32 参考");
        };
        run32("2d_vec", sgemm_2d_vec, cfg2d::THREADS);
        run32("db_reg", sgemm_db_reg, cfg2d::THREADS);
        run32("db_cpasync", sgemm_db_cpasync, cfg2d::THREADS);
        run32("warptile", sgemm_warptile, cfgwt::THREADS);

        for (size_t i = 0; i < nn; ++i) C.p[i] = NAN;
        launch<half>(sgemm_wmma_naive, dim3(n / 32, n / 32), dim3(128), A16.p, B16.p, C.p, n);
        check("wmma_naive", 1e-5, max_err(C.p, ref16), " half 输入参考");
        check("wmma_naive", 1e-2, max_err(C.p, ref), " FP32 参考");

        for (size_t i = 0; i < nn; ++i) C.p[i] = NAN;
        launch<half>(sgemm_wmma_fp16, dim3(n / 128, n / 128), dim3(cfgwm::THREADS), A16.p, B16.p, C.p, n);
        check("wmma_fp16", 1e-5, max_err(C.p, ref16), " half 输入参考");
        check("wmma_fp16", 1e-2, max_err(C.p, ref), " FP32 参考");
    }
    printf(fails ? "\n有 %d 项 WRONG\n" : "\n全部 OK\n", fails);
    return fails ? 1 : 0;
}
