// test_index_emulation.cpp —— 在没有 GPU 的机器上检查 sgemm5.cu 里 4 个 kernel 的下标数学
//
// 两层检查：
//   (A) 真代码 + 线程模拟：把 sgemm5.cu 里「GPU kernels 开始 / 结束」之间的源码原样抽出来，
//       用一组 shim 宏（__global__、__shared__、threadIdx、__syncthreads、float4 ...）在 CPU 上编译。
//       一个 block 的每个 CUDA 线程对应一个 std::thread，__syncthreads 用 std::barrier 实现，
//       block 依次执行。矩阵放在 mmap 出来的页里，前后各留一页 PROT_NONE 保护页，越界读写直接段错误。
//   (B) 逐线程顺序模拟：按同样的下标公式把每个 block、每个 k 轮的「搬运阶段 / 计算阶段 / 写回」
//       分开顺序执行（阶段之间就是 barrier 的位置），对每一个共享内存 / 全局内存下标做 assert 越界检查，
//       并检查每次 float4 访问的下标是 4 的倍数（16 字节对齐）。
//   两层都和 CPU 参考矩阵乘比较，N = 128 和 256。
//
// 构建与运行（g++ 或 clang++，需要 C++20 的 <barrier>）：
//   sed -n '/GPU kernels 开始/,/GPU kernels 结束/p' sgemm5.cu > kernels_extracted.inc
//   g++ -std=c++20 -O1 -pthread test_index_emulation.cpp -o test_index_emulation   # 或 clang++ / zig c++
//   ./test_index_emulation
//   可加 -DSKIP_SECOND_SYNC 或 -DSKIP_FIRST_SYNC 编译 (A)，观察去掉同步后结果出错（不保证每次都错）。

#include <barrier>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

// ---------------------------- CUDA shim ----------------------------
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(x)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
thread_local dim3 threadIdx, blockIdx;
thread_local std::barrier<>* g_bar = nullptr;
inline void __syncthreads() { g_bar->arrive_and_wait(); }

struct __attribute__((aligned(16))) float4 { float x, y, z, w; };
inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }

#ifdef SKIP_FIRST_SYNC
#define SYNC_AFTER_LOAD()
#else
#define SYNC_AFTER_LOAD() __syncthreads()
#endif
#ifdef SKIP_SECOND_SYNC
#define SYNC_AFTER_COMPUTE()
#else
#define SYNC_AFTER_COMPUTE() __syncthreads()
#endif

#include "kernels_extracted.inc"   // sgemm5.cu 里的 kernel 源码，原样

// ---------------------------- 带保护页的矩阵 ----------------------------
struct Guarded {
    void* base = nullptr;
    size_t total = 0;
    float* p = nullptr;
    explicit Guarded(size_t count) {
        const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
        const size_t bytes = count * sizeof(float);
        const size_t body = (bytes + pg - 1) / pg * pg;
        total = body + 2 * pg;
        base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(base != MAP_FAILED);
        mprotect(base, pg, PROT_NONE);                          // 前保护页
        mprotect((char*)base + pg + body, pg, PROT_NONE);       // 后保护页
        p = (float*)((char*)base + pg + body - bytes);          // 数组末尾紧贴后保护页
        assert(((uintptr_t)p % 16) == 0);
    }
    ~Guarded() { munmap(base, total); }
};

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
    for (size_t i = 0; i < ref.size(); ++i) w = std::max(w, std::fabs(x[i] - ref[i]) / (std::fabs(ref[i]) + 1.0));
    return w;
}

using KernelFn = void (*)(const float*, const float*, float*, int);

// (A) 每个 CUDA 线程一个 std::thread，block 依次执行
static void launch_threads(KernelFn k, dim3 grid, dim3 block, const float* A, const float* B, float* C, int n) {
    const unsigned T = block.x * block.y;
    std::barrier<> bar((std::ptrdiff_t)T);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < T; ++t) {
        pool.emplace_back([&, t] {
            g_bar = &bar;
            threadIdx = dim3(t % block.x, t / block.x, 0);
            for (unsigned by = 0; by < grid.y; ++by)
                for (unsigned bx = 0; bx < grid.x; ++bx) {
                    blockIdx = dim3(bx, by, 0);
                    k(A, B, C, n);
                    bar.arrive_and_wait();  // block 结束：保证共享数组（static）被下一个 block 复用前大家都走完了
                }
        });
    }
    for (auto& th : pool) th.join();
}

// ---------------------------- (B) 顺序模拟 + 下标断言 ----------------------------
#define CHK(cond) do { if (!(cond)) { fprintf(stderr, "断言失败 %s 行 %d: %s\n", __FILE__, __LINE__, #cond); std::exit(2); } } while (0)

static void emu_1d(const float* A, const float* B, float* C, int n) {
    const int BM = 64, BN = 64, BK = 8, TM = 8, THREADS = 512;
    std::vector<float> As(BM * BK), Bs(BK * BN);
    for (int cRow = 0; cRow < n / BM; ++cRow)
        for (int cCol = 0; cCol < n / BN; ++cCol) {
            std::vector<float> acc((size_t)THREADS * TM, 0.0f);
            std::vector<int> covA(BM * BK), covB(BK * BN);
            for (int bk = 0; bk < n; bk += BK) {
                std::fill(covA.begin(), covA.end(), 0); std::fill(covB.begin(), covB.end(), 0);
                for (int tid = 0; tid < THREADS; ++tid) {          // 搬运阶段
                    int iRA = tid / BK, iCA = tid % BK, iRB = tid / BN, iCB = tid % BN;
                    int sa = iRA * BK + iCA, sb = iRB * BN + iCB;
                    CHK(sa >= 0 && sa < BM * BK); CHK(sb >= 0 && sb < BK * BN);
                    size_t ga = (size_t)cRow * BM * n + (size_t)iRA * n + bk + iCA;
                    size_t gb = (size_t)cCol * BN + (size_t)(bk + iRB) * n + iCB;
                    CHK(ga < (size_t)n * n); CHK(gb < (size_t)n * n);
                    CHK(iCA + bk < n && iCB + cCol * BN < n);
                    As[sa] = A[ga]; Bs[sb] = B[gb]; covA[sa]++; covB[sb]++;
                }
                for (int c : covA) CHK(c == 1);                     // 每个格子恰好被搬一次
                for (int c : covB) CHK(c == 1);
                for (int tid = 0; tid < THREADS; ++tid) {          // 计算阶段
                    int tc = tid % BN, tr = tid / BN;
                    for (int dot = 0; dot < BK; ++dot) {
                        int ib = dot * BN + tc; CHK(ib < BK * BN);
                        float b = Bs[ib];
                        for (int i = 0; i < TM; ++i) {
                            int ia = (tr * TM + i) * BK + dot; CHK(ia < BM * BK);
                            acc[(size_t)tid * TM + i] += As[ia] * b;
                        }
                    }
                }
            }
            for (int tid = 0; tid < THREADS; ++tid) {
                int tc = tid % BN, tr = tid / BN;
                for (int i = 0; i < TM; ++i) {
                    size_t gc = (size_t)(cRow * BM + tr * TM + i) * n + cCol * BN + tc;
                    CHK(gc < (size_t)n * n);
                    C[gc] = acc[(size_t)tid * TM + i];
                }
            }
        }
}

static void emu_2d(const float* A, const float* B, float* C, int n, bool vec) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, THREADS = 256;
    std::vector<float> As(BM * BK), Bs(BK * BN);
    for (int cRow = 0; cRow < n / BM; ++cRow)
        for (int cCol = 0; cCol < n / BN; ++cCol) {
            std::vector<float> acc((size_t)THREADS * TM * TN, 0.0f);
            std::vector<int> covA(BM * BK), covB(BK * BN);
            std::vector<int> covC((size_t)BM * BN, 0);
            const size_t aBase = (size_t)cRow * BM * n, bBase = (size_t)cCol * BN;
            for (int bk = 0; bk < n; bk += BK) {
                std::fill(covA.begin(), covA.end(), 0); std::fill(covB.begin(), covB.end(), 0);
                for (int tid = 0; tid < THREADS; ++tid) {          // 搬运阶段
                    if (!vec) {
                        int iRA = tid / BK, iCA = tid % BK, sA = THREADS / BK;
                        int iRB = tid / BN, iCB = tid % BN, sB = THREADS / BN;
                        for (int off = 0; off < BM; off += sA) {
                            int s = (iRA + off) * BK + iCA; CHK(s < BM * BK);
                            size_t g = aBase + (size_t)(iRA + off) * n + bk + iCA; CHK(g < (size_t)n * n);
                            As[s] = A[g]; covA[s]++;
                        }
                        for (int off = 0; off < BK; off += sB) {
                            int s = (iRB + off) * BN + iCB; CHK(s < BK * BN);
                            size_t g = bBase + (size_t)(bk + iRB + off) * n + iCB; CHK(g < (size_t)n * n);
                            Bs[s] = B[g]; covB[s]++;
                        }
                    } else {
                        int iRA = tid / (BK / 4), iCA = tid % (BK / 4);
                        int iRB = tid / (BN / 4), iCB = tid % (BN / 4);
                        size_t ga = aBase + (size_t)iRA * n + bk + iCA * 4;
                        CHK(ga % 4 == 0); CHK(ga + 3 < (size_t)n * n);   // float4 对齐 + 不越界
                        for (int q = 0; q < 4; ++q) {
                            int s = (iCA * 4 + q) * BM + iRA; CHK(s < BM * BK);
                            As[s] = A[ga + q]; covA[s]++;
                        }
                        int sb = iRB * BN + iCB * 4; CHK(sb % 4 == 0); CHK(sb + 3 < BK * BN);
                        size_t gb = bBase + (size_t)(bk + iRB) * n + iCB * 4;
                        CHK(gb % 4 == 0); CHK(gb + 3 < (size_t)n * n);
                        for (int q = 0; q < 4; ++q) { Bs[sb + q] = B[gb + q]; covB[sb + q]++; }
                    }
                }
                for (int c : covA) CHK(c == 1);
                for (int c : covB) CHK(c == 1);
                for (int tid = 0; tid < THREADS; ++tid) {          // 计算阶段
                    int tc = tid % (BN / TN), tr = tid / (BN / TN);
                    for (int dot = 0; dot < BK; ++dot) {
                        float rM[8], rN[8];
                        for (int i = 0; i < TM; ++i) {
                            int s = vec ? dot * BM + tr * TM + i : (tr * TM + i) * BK + dot;
                            if (vec && (i % 4 == 0)) CHK(s % 4 == 0);
                            CHK(s < BM * BK); rM[i] = As[s];
                        }
                        for (int j = 0; j < TN; ++j) {
                            int s = dot * BN + tc * TN + j;
                            if (vec && (j % 4 == 0)) CHK(s % 4 == 0);
                            CHK(s < BK * BN); rN[j] = Bs[s];
                        }
                        for (int i = 0; i < TM; ++i)
                            for (int j = 0; j < TN; ++j) acc[(size_t)tid * 64 + i * TN + j] += rM[i] * rN[j];
                    }
                }
            }
            for (int tid = 0; tid < THREADS; ++tid) {
                int tc = tid % (BN / TN), tr = tid / (BN / TN);
                for (int i = 0; i < TM; ++i) {
                    size_t g0 = (size_t)(cRow * BM + tr * TM + i) * n + cCol * BN + tc * TN;
                    if (vec) CHK(g0 % 4 == 0);
                    for (int j = 0; j < TN; ++j) {
                        CHK(g0 + j < (size_t)n * n);
                        C[g0 + j] = acc[(size_t)tid * 64 + i * TN + j];
                        covC[(size_t)(tr * TM + i) * BN + tc * TN + j]++;
                    }
                }
            }
            for (int c : covC) CHK(c == 1);                         // C 小块每个格子恰好写一次
        }
}

int main() {
    int fails = 0;
    for (int n : {128, 256}) {
        const size_t nn = (size_t)n * n;
        Guarded gA(nn), gB(nn), gC(nn);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> d(0.f, 1.f);
        for (size_t i = 0; i < nn; ++i) { gA.p[i] = d(gen); gB.p[i] = d(gen); }
        std::vector<double> ref(nn);
        ref_matmul(gA.p, gB.p, ref.data(), n);

        struct Case { const char* name; KernelFn k; dim3 grid, block; };
        Case cases[] = {
            {"tile16", sgemm_tile16, dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16)},
            {"1d_tm8", sgemm_1d_tm8, dim3(n / 64, n / 64), dim3(512)},
            {"2d_8x8", sgemm_2d_8x8, dim3(n / 128, n / 128), dim3(256)},
            {"2d_vec", sgemm_2d_vec, dim3(n / 128, n / 128), dim3(256)},
        };
        for (auto& c : cases) {
            for (size_t i = 0; i < nn; ++i) gC.p[i] = NAN;      // 没写到的格子会是 NaN → 误差变 NaN → 失败
            launch_threads(c.k, c.grid, c.block, gA.p, gB.p, gC.p, n);
            double e = max_err(gC.p, ref);
            bool ok = e < 1e-4;
            fails += !ok;
            printf("(A) 真代码线程模拟  N=%4d %-8s max_rel_err=%.2e %s\n", n, c.name, e, ok ? "PASS" : "FAIL");
        }
        auto runB = [&](const char* name, std::function<void()> f) {
            for (size_t i = 0; i < nn; ++i) gC.p[i] = NAN;
            f();
            double e = max_err(gC.p, ref);
            bool ok = e < 1e-4;
            fails += !ok;
            printf("(B) 顺序模拟+断言    N=%4d %-8s max_rel_err=%.2e %s\n", n, name, e, ok ? "PASS" : "FAIL");
        };
        runB("1d_tm8", [&] { emu_1d(gA.p, gB.p, gC.p, n); });
        runB("2d_8x8", [&] { emu_2d(gA.p, gB.p, gC.p, n, false); });
        runB("2d_vec", [&] { emu_2d(gA.p, gB.p, gC.p, n, true); });
    }
    printf(fails ? "\n有 %d 项失败\n" : "\n全部通过\n", fails);
    return fails ? 1 : 0;
}
