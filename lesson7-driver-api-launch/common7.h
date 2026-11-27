// common7.h —— 第 7 课的小工具：检查 driver API / runtime API 的返回值、主机端计时
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>  // driver API：cu 开头的函数，库是 libcuda.so（显卡驱动自带）

// driver API 的每个函数都返回 CUresult。出错时打印「名字 + 说明 + 哪一行」然后退出。
// cuGetErrorName / cuGetErrorString 本身也是 driver API，cuInit 之前也能调用。
#define CU_CHECK(call)                                                          \
    do {                                                                        \
        CUresult _r = (call);                                                   \
        if (_r != CUDA_SUCCESS) {                                               \
            const char *_name = nullptr, *_str = nullptr;                       \
            cuGetErrorName(_r, &_name);                                         \
            cuGetErrorString(_r, &_str);                                        \
            std::fprintf(stderr, "[driver] %s:%d  %s 返回 %d (%s): %s\n",       \
                         __FILE__, __LINE__, #call, (int)_r,                    \
                         _name ? _name : "?", _str ? _str : "?");               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// runtime API（cuda 开头，库是 libcudart）的检查，只在 .cu 文件里用
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[runtime] %s:%d  %s 返回 %d (%s): %s\n",      \
                         __FILE__, __LINE__, #call, (int)_e,                    \
                         cudaGetErrorName(_e), cudaGetErrorString(_e));         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)
#endif

// 主机端计时（微秒）。第 7 课量的都是「CPU 这边花了多久」，所以不用 cudaEvent。
inline double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}
