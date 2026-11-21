// omp_matmul.cpp —— 矩阵乘 C = A * B：循环顺序 + OpenMP 并行，测时间和 GFLOPS
// 编译：g++ -O2 -std=c++17 -fopenmp omp_matmul.cpp -o omp_matmul
// 运行：OMP_NUM_THREADS=4 ./omp_matmul [N] [skip_ijk]
//       N 默认 1024；第二个参数写 1 表示跳过很慢的串行 i-j-k（N 很大时用）
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common.h"

// 版本 1：教科书写法 i-j-k（内层循环按列读 B，对缓存不友好）
void matmul_ijk(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];  // B 跳着读
            C[i * n + j] = sum;
        }
}

// 版本 2：并行 i-j-k —— 按行 i 分给各线程
void matmul_ijk_omp(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

// 版本 3：并行 i-k-j —— 内层循环连续读 B 的一行、连续写 C 的一行
void matmul_ikj_omp(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float* c = C + (size_t)i * n;               // 第 i 行的起点（每个线程只写自己的行）
        for (int j = 0; j < n; ++j) c[j] = 0.0f;
        for (int k = 0; k < n; ++k) {
            const float a = A[(size_t)i * n + k];   // 在寄存器里拿着 A[i][k]
            const float* b = B + (size_t)k * n;     // B 的第 k 行
            for (int j = 0; j < n; ++j) c[j] += a * b[j];
        }
    }
}

// 计时：跑 reps 次取最快的一次（排除偶尔被系统打断的干扰）
template <typename F>
double best_time(F f, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t = omp_get_wtime();
        f();
        t = omp_get_wtime() - t;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 1024;
    const bool skip_ijk = (argc > 2) && atoi(argv[2]) != 0;
    const size_t nn = (size_t)n * n;
    std::vector<float> A(nn), B(nn), C(nn), Ref(nn);
    fill_random(A, 1);
    fill_random(B, 2);
    printf("N = %d, threads = %d\n", n, omp_get_max_threads());
    printf("%-18s %10s %9s %12s\n", "version", "time(ms)", "GFLOPS", "max_rel_err");

    // 串行 i-k-j：既是基准（算加速比的分母），也是标准答案
    double t_ref = best_time([&] { matmul_ikj_serial(A.data(), B.data(), Ref.data(), n); }, 3);
    printf("%-18s %10.1f %9.2f %12s\n", "serial ikj", t_ref * 1e3, gflops(n, t_ref), "(reference)");

    bool all_ok = true;
    auto report = [&](const char* name, double t) {
        double err = max_rel_err(C.data(), Ref.data(), nn);
        bool ok = err < 1e-3;            // float 求和顺序不同会有微小误差，1e-3 足够宽松
        all_ok = all_ok && ok;
        printf("%-18s %10.1f %9.2f %12.2e  %s\n", name, t * 1e3, gflops(n, t), err, ok ? "OK" : "WRONG");
    };

    if (!skip_ijk) {
        double t1 = best_time([&] { matmul_ijk(A.data(), B.data(), C.data(), n); }, 1);
        report("serial ijk", t1);
        double t2 = best_time([&] { matmul_ijk_omp(A.data(), B.data(), C.data(), n); }, 3);
        report("omp ijk", t2);
    }
    double t3 = best_time([&] { matmul_ikj_omp(A.data(), B.data(), C.data(), n); }, 3);
    report("omp ikj", t3);

    printf("speedup omp ikj vs serial ikj: %.2fx  (efficiency %.0f%% with %d threads)\n",
           t_ref / t3, 100.0 * t_ref / t3 / omp_get_max_threads(), omp_get_max_threads());
    return all_ok ? 0 : 1;
}
