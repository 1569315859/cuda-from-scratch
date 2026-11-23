// cache_order.cpp —— 第 4 课第 7 章：用 cachegrind / perf 亲眼看第 3 课「循环顺序」的缓存差别
// 用法：./cache_order ijk|ikj [N]     （N 默认 256；cachegrind 很慢，别用太大的 N）
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common.h"

// 教科书写法：最内层 k，B[k][j] 每次跳一整行（n*4 字节）
__attribute__((noinline)) static void matmul_ijk(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

// 包一层 noinline：让 cg_annotate 的函数统计里单独出现 run_ikj，而不是被并进 main
__attribute__((noinline)) static void run_ikj(const float* A, const float* B, float* C, int n) {
    matmul_ikj_serial(A, B, C, n);
}

int main(int argc, char** argv) {
    if (argc < 2 || (std::strcmp(argv[1], "ijk") != 0 && std::strcmp(argv[1], "ikj") != 0)) {
        fprintf(stderr, "用法: %s ijk|ikj [N]\n", argv[0]);
        return 1;
    }
    const bool ijk = std::strcmp(argv[1], "ijk") == 0;
    const int n = argc > 2 ? std::atoi(argv[2]) : 256;
    const size_t nn = (size_t)n * n;

    std::vector<float> A(nn), B(nn), C(nn);
    fill_random(A, 1);
    fill_random(B, 2);

    auto t0 = std::chrono::steady_clock::now();
    if (ijk) matmul_ijk(A.data(), B.data(), C.data(), n);
    else     run_ikj(A.data(), B.data(), C.data(), n);   // 调用 common.h 里的 i-k-j
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();

    double checksum = 0.0;  // 打印校验和：防止编译器把整个计算当成「没用」优化掉
    for (size_t i = 0; i < nn; ++i) checksum += C[i];
    printf("%s N=%d  time=%.3f ms  GFLOPS=%.2f  checksum=%.6e\n", argv[1], n, s * 1e3,
           gflops(n, s), checksum);
    return 0;
}
