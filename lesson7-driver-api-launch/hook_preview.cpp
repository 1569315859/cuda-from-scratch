// hook_preview.cpp —— 实验 6（第 10 课的预告）：用 LD_PRELOAD 拦截 cuLaunchKernel，数一数被调用了几次。
// 编译：g++ -O2 -shared -fPIC hook_preview.cpp -o libhook_preview.so -ldl -I$(CUDA_HOME)/include
// 运行：LD_PRELOAD=./libhook_preview.so ./drv_vadd vadd.cubin      （预测：能数到 2 次）
//       LD_PRELOAD=./libhook_preview.so ./launch_overhead            （预测：正好 20000 次，全部来自 T4(c)；<<<>>> 和 cudaLaunchKernel 一次都没拦到 —— 为什么？第 7.3 节）
//
// 原理：动态链接器先在 LD_PRELOAD 的库里找符号。程序里调用 cuLaunchKernel 时，
// 会先找到这里的同名函数；我们计数，再用 dlsym(RTLD_NEXT, ...) 找到「下一个」—— 真正的 libcuda 里的实现，转过去。
#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <cuda.h>

static std::atomic<long> g_count{0};

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                           unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                           CUstream s, void** params, void** extra) {
    using Fn = CUresult(CUDAAPI*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                  unsigned, CUstream, void**, void**);
    static Fn real = (Fn)dlsym(RTLD_NEXT, "cuLaunchKernel");
    if (g_count.fetch_add(1) < 3)
        std::fprintf(stderr, "[hook] cuLaunchKernel grid=(%u,%u,%u) block=(%u,%u,%u)\n", gx, gy, gz, bx, by, bz);
    return real(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
}

__attribute__((destructor)) static void report() {
    std::fprintf(stderr, "[hook] 进程结束：一共拦截到 cuLaunchKernel %ld 次\n", g_count.load());
}
