// 第 2 课：并行归约（求和）——从最慢到接近带宽上限的 5 个版本
#include "common.cuh"
#include <vector>
#include <chrono>

// ---------- v0：每个线程直接原子加到同一个全局变量 ----------
__global__ void reduce_v0_atomic(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);              // 1600 万个线程抢同一个地址
}

// ---------- v1：共享内存 + 交错配对（有 warp 分化） ----------
__global__ void reduce_v1_interleaved(const float* in, float* out, int n) {
    extern __shared__ float s[];                   // 大小在启动时第 3 个参数给出
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;               // 每个线程搬一个数到共享内存
    __syncthreads();                               // 等全班都搬完
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)               // 只有 0,2,4.. 然后 0,4,8.. 干活
            s[tid] += s[tid + stride];
        __syncthreads();                           // 每一轮结束都要等齐
    }
    if (tid == 0) atomicAdd(out, s[0]);            // 每个 block 只原子加一次
}

// ---------- v2：共享内存 + 连续配对（消除分化） ----------
__global__ void reduce_v2_sequential(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)                          // 前一半线程干活，后一半休息
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ---------- v3：搬数据时先加一次（一个线程管两个数） ----------
__global__ void reduce_v3_add_on_load(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;   // 每个 block 负责 2*blockDim 个数
    float v = 0.0f;
    if (i < n) v = in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ---------- v4：grid-stride 循环 + warp shuffle ----------
__device__ float warp_sum(float v) {
    // 在一个 warp 的 32 个线程之间直接交换寄存器，不经过共享内存
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;                                      // lane 0 拿到 32 个数的和
}

__global__ void reduce_v4_shuffle(const float* in, float* out, int n) {
    float v = 0.0f;
    // 线程数比元素少：每个线程按“全体线程数”为步长，循环累加多个元素
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        v += in[i];
    v = warp_sum(v);                               // 第 1 步：warp 内求和

    __shared__ float warp_sums[32];                // 一个 block 最多 1024/32 = 32 个 warp
    int lane = threadIdx.x % 32;                   // 我在 warp 里是第几个
    int wid  = threadIdx.x / 32;                   // 我是第几个 warp
    if (lane == 0) warp_sums[wid] = v;             // 每个 warp 的代表写一个数
    __syncthreads();

    if (wid == 0) {                                // 第 2 步：第 0 个 warp 把各 warp 的和再加起来
        int num_warps = blockDim.x / 32;
        v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------- 计时工具：热身 1 次 + 正式跑 R 次取平均 ----------
template <typename F>
void bench(const char* name, F launch, float* d_out, float expect, size_t bytes, int R = 10) {
    cudaEvent_t s, t;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&t));

    float h_out = 0.0f;
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));        // 结果清零
    launch();                                               // 热身，同时拿结果检查
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    float total = 0.0f;
    for (int r = 0; r < R; r++) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(s));
        launch();
        CUDA_CHECK(cudaEventRecord(t));
        CUDA_CHECK(cudaEventSynchronize(t));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, t));
        total += ms;
    }
    float avg = total / R;
    double gbps = bytes / (avg * 1e-3) / 1e9;               // 只读一遍输入
    printf("%-24s %9.3f ms %8.1f GB/s   sum=%.0f  %s\n",
           name, avg, gbps, h_out, h_out == expect ? "OK" : "WRONG");
    cudaEventDestroy(s);
    cudaEventDestroy(t);
}

int main() {
    const int n = 1 << 24;                         // 16777216 个数
    const size_t bytes = n * sizeof(float);
    // 全部填 1.0：正确答案正好是 n，而且 float 能精确表示，方便判断对错
    std::vector<float> h(n, 1.0f);

    auto c0 = std::chrono::high_resolution_clock::now();
    double cpu_sum = 0.0;
    for (int i = 0; i < n; i++) cpu_sum += h[i];
    auto c1 = std::chrono::high_resolution_clock::now();
    printf("%-24s %9.3f ms                sum=%.0f\n", "CPU loop",
           std::chrono::duration<double, std::milli>(c1 - c0).count(), cpu_sum);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    int sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    const size_t smem = block * sizeof(float);     // 动态共享内存大小
    const float expect = (float)n;

    bench("v0 atomic", [&] { reduce_v0_atomic<<<grid, block>>>(d_in, d_out, n); },
          d_out, expect, bytes, 3);                 // 太慢，只跑 3 次
    bench("v1 interleaved", [&] { reduce_v1_interleaved<<<grid, block, smem>>>(d_in, d_out, n); },
          d_out, expect, bytes);
    bench("v2 sequential", [&] { reduce_v2_sequential<<<grid, block, smem>>>(d_in, d_out, n); },
          d_out, expect, bytes);
    int grid3 = (n + block * 2 - 1) / (block * 2);
    bench("v3 add-on-load", [&] { reduce_v3_add_on_load<<<grid3, block, smem>>>(d_in, d_out, n); },
          d_out, expect, bytes);
    int grid4 = sm * 8;                            // 只开“SM 数 × 8”个 block
    bench("v4 shuffle", [&] { reduce_v4_shuffle<<<grid4, block>>>(d_in, d_out, n); },
          d_out, expect, bytes);

    printf("SM count = %d, v4 grid = %d blocks\n", sm, grid4);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
