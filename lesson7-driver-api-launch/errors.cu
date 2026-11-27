// errors.cu —— 实验 4：CUDA 的两种错误（第 6 章）
//   非粘性（non-sticky）：参数错了，启动根本没发出去。读一次 cudaGetLastError 就清掉，context 还能用。
//   粘性（sticky）：kernel 在 GPU 上真的出事了（越界访问等）。context 坏掉，之后所有调用都返回同一个错，
//                  只能 cudaDeviceReset（或进程退出）重建 primary context。
// 编译：nvcc -O2 -arch=sm_86 errors.cu -o errors -lcuda

#include "common7.h"

__global__ void empty_kernel() {}
__global__ void bad_write(float* p, long long offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[offset] = 1.0f;
}

static void show(const char* what, cudaError_t e) {
    std::printf("  %-48s → %s\n", what, cudaGetErrorName(e));
}

int main() {
    CUDA_CHECK(cudaFree(0));

    std::printf("[A] 非粘性错误：一个 block 2048 个线程（上限是 1024）\n");
    empty_kernel<<<1, 2048>>>();
    show("cudaPeekAtLastError()（只看不清）", cudaPeekAtLastError());
    show("cudaPeekAtLastError() 再看一次", cudaPeekAtLastError());
    show("cudaGetLastError()（读完就清零）", cudaGetLastError());
    show("cudaGetLastError() 再读一次", cudaGetLastError());
    float* p = nullptr;
    show("之后 cudaMalloc 还能用吗", cudaMalloc(&p, 1024 * sizeof(float)));

    std::printf("[B] 粘性错误：kernel 往离 p 很远（256 GB）的地址写\n");
    bad_write<<<1, 32>>>(p, 1LL << 36);
    show("启动之后马上 cudaGetLastError()", cudaGetLastError());   // 预测：cudaSuccess，GPU 还没跑到
    show("cudaDeviceSynchronize()", cudaDeviceSynchronize());      // 预测：cudaErrorIllegalAddress
    show("cudaGetLastError()", cudaGetLastError());
    show("cudaGetLastError() 再读一次（清得掉吗）", cudaGetLastError());
    float* q = nullptr;
    show("之后 cudaMalloc", cudaMalloc(&q, 1024 * sizeof(float)));
    empty_kernel<<<1, 1>>>();
    show("之后再启动一个正常的 kernel 再同步", cudaDeviceSynchronize());
    CUresult r = cuCtxSynchronize();
    const char* rn = nullptr; cuGetErrorName(r, &rn);
    std::printf("  %-48s → %s\n", "driver 层 cuCtxSynchronize()", rn);

    std::printf("[C] cudaDeviceReset() 销毁 primary context，下次调用时重建\n");
    show("cudaDeviceReset()", cudaDeviceReset());
    show("之后 cudaMalloc", cudaMalloc(&q, 1024 * sizeof(float)));
    empty_kernel<<<1, 1>>>();
    show("之后启动正常 kernel 再同步", cudaDeviceSynchronize());
    // 注意：Reset 之前分配的 p 已经随旧 context 一起没了，不能再 cudaFree(p)
    CUDA_CHECK(cudaFree(q));
    return 0;
}
