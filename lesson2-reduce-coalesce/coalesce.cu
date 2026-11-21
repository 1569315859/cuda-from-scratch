// 第 2 课：合并访问 vs 跳着访问
#include "common.cuh"
#include <vector>

// S = 1：相邻线程访问相邻地址（合并访问）
// S > 1：相邻线程访问的地址相隔 n/S 个元素（跳着访问）
// 这个映射是一一对应的，每个元素正好被拷贝一次
__global__ void copy_pattern(const float* in, float* out, int n, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i % S) * (n / S) + i / S;
        out[j] = in[j];
    }
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = n * sizeof(float);
    std::vector<float> h(n, 1.0f);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t s, t;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&t));
    const int block = 256, grid = (n + block - 1) / block;

    int strides[] = {1, 2, 4, 8, 16, 32};
    for (int S : strides) {
        copy_pattern<<<grid, block>>>(d_in, d_out, n, S);   // 热身
        CUDA_CHECK(cudaGetLastError());
        float total = 0.0f;
        for (int r = 0; r < 10; r++) {
            CUDA_CHECK(cudaEventRecord(s));
            copy_pattern<<<grid, block>>>(d_in, d_out, n, S);
            CUDA_CHECK(cudaEventRecord(t));
            CUDA_CHECK(cudaEventSynchronize(t));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, t));
            total += ms;
        }
        float avg = total / 10;
        printf("S = %2d : %7.3f ms  %6.1f GB/s\n", S, avg, 2.0 * bytes / (avg * 1e-3) / 1e9);
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
