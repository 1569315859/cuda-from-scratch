// common.h —— 第 5 课原样沿用第 4 课（最早来自第 3 课）的 CPU 小工具：填随机数、串行 i-k-j 矩阵乘（标准答案）、误差、GFLOPS
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// 用固定种子填 [0,1) 的随机数，每次运行数据都一样，方便对比
inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// 串行 i-k-j 矩阵乘：C = A * B，都是 n x n、按行存储（row-major）
// 这是正确性检查用的「标准答案」
inline void matmul_ikj_serial(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = 0.0f;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const float a = A[i * n + k];
            for (int j = 0; j < n; ++j) C[i * n + j] += a * B[k * n + j];
        }
}

// 返回两个数组的最大相对误差（分母加 1，避免除以接近 0 的数）
inline double max_rel_err(const float* x, const float* ref, size_t count) {
    double worst = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double e = std::fabs((double)x[i] - (double)ref[i]) / (std::fabs((double)ref[i]) + 1.0);
        if (e > worst) worst = e;
    }
    return worst;
}

// 矩阵乘的计算量：n^3 次乘法 + n^3 次加法 = 2n^3 次浮点运算
inline double gflops(int n, double seconds) {
    return 2.0 * (double)n * n * n / seconds / 1e9;
}
