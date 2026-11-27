// proc_addr.cu —— 实验 5：同一个 driver 函数，三种拿法拿到的地址一样吗？（第 7 章，为第 10 课的 hook 做准备）
//   (1) dlsym：在 libcuda.so.1 的导出符号表里按名字找
//   (2) cuGetProcAddress：问驱动「这个名字在 CUDA x.y 的语义下对应哪个实现」
//   (3) cudaGetDriverEntryPointByVersion：runtime 版本的 (2)
// 重点看 cuMemAlloc：头文件把它 #define 成了 cuMemAlloc_v2，导出表里两个名字都有，指向的却不是同一个函数。
// 编译：nvcc -O2 -arch=sm_86 proc_addr.cu -o proc_addr -lcuda -ldl

#include <dlfcn.h>
#include "common7.h"

// 用 #undef 拿回「原始名字」的字符串，避免被 cuda.h 的宏改掉
static void probe(void* lib, const char* name) {
    void* by_dlsym = dlsym(lib, name);
    void* by_proc = nullptr;
    CUdriverProcAddressQueryResult st;
    CUresult r = cuGetProcAddress(name, &by_proc, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &st);
    void* by_rt = nullptr;
    cudaDriverEntryPointQueryResult rst;
    cudaError_t e = cudaGetDriverEntryPointByVersion(name, &by_rt, CUDART_VERSION, cudaEnableDefault, &rst);
    std::printf("%-22s dlsym=%-16p procaddr=%-16p runtime=%-16p %s\n", name, by_dlsym,
                r == CUDA_SUCCESS ? by_proc : nullptr, e == cudaSuccess ? by_rt : nullptr,
                (by_dlsym == by_proc) ? "dlsym 与 procaddr 相同" : "dlsym 与 procaddr 不同！");
}

int main() {
    CU_CHECK(cuInit(0));
    CUDA_CHECK(cudaFree(0));
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);  // 已经加载过了，只拿句柄
    if (!lib) { std::printf("dlopen 失败: %s\n", dlerror()); return 1; }
    std::printf("编译时 CUDA_VERSION = %d, CUDART_VERSION = %d\n", CUDA_VERSION, CUDART_VERSION);
    probe(lib, "cuLaunchKernel");
    probe(lib, "cuLaunchKernel_ptsz");
    probe(lib, "cuMemAlloc");
    probe(lib, "cuMemAlloc_v2");
    probe(lib, "cuCtxCreate");
    probe(lib, "cuCtxCreate_v4");
    probe(lib, "cuMemcpyHtoD_v2");
    return 0;
}
