#include "common.cuh"
#include <vector>
#include <chrono>
#include <cmath>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    if (i < n) c[i] = a[i] + b[i];                  // 边界检查
}

int main() {
    const int n = 1 << 24;                 // 16M 个 float，约 64 MB
    const size_t bytes = n * sizeof(float);
    std::vector<float> ha(n), hb(n), hc(n), ref(n);
    for (int i = 0; i < n; i++) { ha[i] = i * 0.5f; hb[i] = i * 0.25f; }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) ref[i] = ha[i] + hb[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));   // 第一次 CUDA 调用会顺带初始化上下文
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    int block = 256;
    int grid = (n + block - 1) / block;    // 向上取整
    vec_add<<<grid, block>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());        // 检查启动参数错误
    CUDA_CHECK(cudaEventRecord(e2));

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));

    float h2d, kern, d2h;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&kern, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, e2, e3));

    int bad = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(hc[i] - ref[i]) > 1e-5f * fabsf(ref[i]) + 1e-6f) bad++;

    double gbps = 3.0 * bytes / (kern * 1e-3) / 1e9;   // 读 a、b，写 c
    printf("CPU loop      : %.2f ms\n", cpu_ms);
    printf("H2D copy      : %.2f ms\n", h2d);
    printf("kernel        : %.3f ms  (%.1f GB/s)\n", kern, gbps);
    printf("D2H copy      : %.2f ms\n", d2h);
    printf("result        : %s (%d mismatches)\n", bad ? "WRONG" : "OK", bad);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}
