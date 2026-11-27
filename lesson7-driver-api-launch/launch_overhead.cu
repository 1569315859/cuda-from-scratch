// launch_overhead.cu —— 实验 3：一次 kernel 启动到底要多久？（第 5 章）
//
// 全部用「什么也不干」的 kernel，这样量到的就是启动本身的开销，不是计算时间。
//   T0  runtime 初始化（cudaFree(0)）              —— 和 CUDA_MODULE_LOADING=EAGER/LAZY 有关
//   T1  某个 kernel 的第一次启动                     —— LAZY 模式下这里才真正加载这个 kernel
//   T2  连续异步启动 20000 次：CPU 每次花多久         —— 前面快、后面慢（队列满了要等 GPU），第 5.4 节
//   T3  每次启动都等它做完：一次「来回」多久          —— 启动延迟 + 同步延迟
//   T4  三种写法比较：<<<>>> / cudaLaunchKernel / driver cuLaunchKernel
//   T5  空 kernel 但 grid 很大：GPU 分发 block 的开销
//
// 编译：nvcc -O2 -arch=sm_86 launch_overhead.cu -o launch_overhead -lcuda
// 运行：make overhead（会分别用 LAZY 和 EAGER 各跑一次）

#include <algorithm>
#include <vector>
#include "common7.h"

__global__ void empty_kernel() {}
__global__ void never_used_before() {}   // 只在 T1 用一次
__global__ void unused_1(float* p) { p[threadIdx.x] *= 2.0f; }  // 从来不启动；EAGER 模式下也会被加载
__global__ void unused_2(float* p) { p[threadIdx.x] += 1.0f; }

static double avg(const std::vector<double>& v, size_t from, size_t to) {
    double s = 0; for (size_t i = from; i < to; ++i) s += v[i]; return s / (double)(to - from);
}

int main() {
    double t0, t1;
    const char* lm = std::getenv("CUDA_MODULE_LOADING");
    std::printf("CUDA_MODULE_LOADING = %s\n", lm ? lm : "(未设置，CUDA 12.2 起默认 LAZY)");

    // ---- T0 ----
    t0 = now_us();
    CUDA_CHECK(cudaFree(0));
    t1 = now_us();
    std::printf("T0 runtime 初始化 cudaFree(0)           %10.1f us\n", t1 - t0);

    // ---- T1 ----
    t0 = now_us();
    never_used_before<<<1, 1>>>();
    t1 = now_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double t2 = now_us();
    std::printf("T1 第一次启动 never_used_before：<<<>>> 返回 %8.1f us，到做完 %8.1f us\n", t1 - t0, t2 - t0);
    t0 = now_us();
    never_used_before<<<1, 1>>>();
    t1 = now_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    t2 = now_us();
    std::printf("   第二次启动同一个 kernel：     返回 %8.1f us，到做完 %8.1f us\n", t1 - t0, t2 - t0);

    // 预热 empty_kernel
    for (int i = 0; i < 100; ++i) empty_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- T2 ----
    const int K = 20000;
    std::vector<double> per(K);
    double tall0 = now_us();
    for (int i = 0; i < K; ++i) {
        double a = now_us();
        empty_kernel<<<1, 1>>>();
        per[i] = now_us() - a;
    }
    double tsub = now_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double tall1 = now_us();
    std::printf("T2 连续异步启动 %d 次：\n", K);
    std::printf("   CPU 每次启动平均（前 500 次）     %8.2f us\n", avg(per, 0, 500));
    std::printf("   CPU 每次启动平均（最后 500 次）   %8.2f us\n", avg(per, K - 500, K));
    std::vector<double> sorted = per; std::sort(sorted.begin(), sorted.end());
    std::printf("   CPU 每次启动 中位数 / 99%% 分位   %8.2f / %8.2f us\n", sorted[K / 2], sorted[(size_t)(K * 0.99)]);
    std::printf("   全部提交完 %.1f us，再等 GPU 做完 %.1f us；GPU 吞吐 = 每个 kernel %.2f us\n",
                tsub - tall0, tall1 - tsub, (tall1 - tall0) / K);

    // ---- T3 ----
    const int R = 2000;
    t0 = now_us();
    for (int i = 0; i < R; ++i) {
        empty_kernel<<<1, 1>>>();
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
    t1 = now_us();
    std::printf("T3 启动 + 等做完，一次来回          %8.2f us（%d 次平均）\n", (t1 - t0) / R, R);

    // ---- T4 ----
    // (a) <<<>>>：编译器把它改写成 cudaLaunchKernel（第 5.1 节）
    t0 = now_us();
    for (int i = 0; i < K; ++i) empty_kernel<<<1, 1>>>();
    t1 = now_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double a_us = (t1 - t0) / K;
    // (b) 直接调用 cudaLaunchKernel
    t0 = now_us();
    for (int i = 0; i < K; ++i)
        CUDA_CHECK(cudaLaunchKernel((const void*)empty_kernel, dim3(1), dim3(1), nullptr, 0, 0));
    t1 = now_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double b_us = (t1 - t0) / K;
    // (c) 从 runtime 的 kernel 拿到 driver 的 CUfunction，直接 cuLaunchKernel
    cudaFunction_t rf;
    CUDA_CHECK(cudaGetFuncBySymbol(&rf, (const void*)empty_kernel));
    CUfunction df = (CUfunction)rf;  // cudaFunction_t 和 CUfunction 是同一种东西
    t0 = now_us();
    for (int i = 0; i < K; ++i)
        CU_CHECK(cuLaunchKernel(df, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
    t1 = now_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double c_us = (t1 - t0) / K;
    std::printf("T4 CPU 提交 %d 次的平均：<<<>>> %.2f us | cudaLaunchKernel %.2f us | cuLaunchKernel %.2f us\n",
                K, a_us, b_us, c_us);

    // ---- T5 ----
    int grids[] = {1, 16, 256, 4096, 65536};
    for (int g : grids) {
        empty_kernel<<<g, 256>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 100; ++i) empty_kernel<<<g, 256>>>();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("T5 空 kernel grid=%6d × 256 线程：GPU 上每个 %8.2f us\n", g, ms * 1000.0f / 100);
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
    }
    return 0;
}
