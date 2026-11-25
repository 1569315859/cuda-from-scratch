// common.cuh —— CUDA 小工具：检查每一个 CUDA / cuBLAS 调用的返回值
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 任何 CUDA API 返回非 cudaSuccess，立刻打印出错的文件和行号并退出
#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

#ifdef USE_CUBLAS
#include <cublas_v2.h>
// cuBLAS 的返回值类型是 cublasStatus_t，单独检查
#define CUBLAS_CHECK(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error: %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
    exit(1); } } while (0)
#endif
