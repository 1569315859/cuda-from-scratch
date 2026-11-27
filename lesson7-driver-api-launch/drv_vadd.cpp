// drv_vadd.cpp —— 实验 1：只用 driver API（cu 开头）做一次向量加法，不用 <<<>>>、不链接 libcudart。
// 这是一个普通的 C++ 文件，用 g++ 就能编译，只需要 cuda.h 和 -lcuda。
//
// 用法：
//   ./drv_vadd vadd.cubin            用 cuModuleLoad 加载 sm_86 机器码
//   ./drv_vadd vadd.ptx              用 cuModuleLoad 加载 PTX（会触发 JIT 编译，第 3 章）
//   ./drv_vadd vadd.fatbin           加载 fatbin，驱动自己挑一份
//   ./drv_vadd vadd.cubin library    改用 CUDA 12 起的 cuLibraryLoadFromFile（和 context 无关的加载，第 4.4 节）
//
// 每一步都打印主机端耗时，第 9 章的表 1、表 2 就填这些数。

#include <cstring>
#include <vector>
#include "common7.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "vadd.cubin";
    const bool use_library = argc > 2 && std::strcmp(argv[2], "library") == 0;
    const int n = 1 << 20;  // 1M 个 float
    double t0, t1;

    // ---- 第 1 步：cuInit。整个进程里第一个必须调用的 driver API（除了查错误名之类的） ----
    // 它打开 /dev/nvidia*（原生 Linux）或 /dev/dxg（WSL2），和内核驱动握手，枚举显卡。
    t0 = now_us();
    CU_CHECK(cuInit(0));
    t1 = now_us();
    std::printf("cuInit                      %10.1f us\n", t1 - t0);

    int drv_ver = 0;
    CU_CHECK(cuDriverGetVersion(&drv_ver));  // 例如 13050 表示驱动支持到 CUDA 13.5
    std::printf("驱动支持的 CUDA 版本         %d.%d\n", drv_ver / 1000, (drv_ver % 1000) / 10);

    // ---- 第 2 步：拿到设备句柄，查几个属性 ----
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    char name[256];
    CU_CHECK(cuDeviceGetName(name, sizeof(name), dev));
    int major = 0, minor = 0, sms = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    std::printf("设备 0: %s  sm_%d%d  %d 个 SM\n", name, major, minor, sms);

    // ---- 第 3 步：创建并「压栈」一个 context ----
    // context = 这个进程在这块 GPU 上的「工作空间」：自己的虚拟地址空间、加载的模块、stream……
    // 这里用的是 primary context（每块卡一个，runtime 也用它），所以和 runtime 混用不会冲突（第 2 章）。
    CUcontext ctx;
    t0 = now_us();
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));  // 引用计数 +1，第一次调用时真正创建
    CU_CHECK(cuCtxSetCurrent(ctx));                // 设为当前线程的 current context
    t1 = now_us();
    std::printf("primary context 创建+设当前  %10.1f us\n", t1 - t0);

    CUmoduleLoadingMode mode;
    CU_CHECK(cuModuleGetLoadingMode(&mode));
    std::printf("模块加载模式                 %s\n",
                mode == CU_MODULE_LAZY_LOADING ? "LAZY（用到哪个 kernel 才加载哪个）" : "EAGER（一次全部加载）");

    // ---- 第 4 步：加载 GPU 代码，拿到 kernel 的句柄 ----
    CUmodule mod = nullptr;
    CUlibrary lib = nullptr;
    CUfunction f_vadd;
    t0 = now_us();
    if (!use_library) {
        CU_CHECK(cuModuleLoad(&mod, path));                      // 加载到「当前 context」
        CU_CHECK(cuModuleGetFunction(&f_vadd, mod, "vadd"));      // 按名字找 kernel
    } else {
        // CUDA 12.0 起：library 不属于任何 context，可以在多个 context 里用（第 4.4 节）
        CU_CHECK(cuLibraryLoadFromFile(&lib, path, nullptr, nullptr, 0, nullptr, nullptr, 0));
        CUkernel k;
        CU_CHECK(cuLibraryGetKernel(&k, lib, "vadd"));
        CU_CHECK(cuKernelGetFunction(&f_vadd, k));                // 取它在当前 context 里的 CUfunction
    }
    t1 = now_us();
    std::printf("加载 %-22s %10.1f us   (%s)\n", path, t1 - t0, use_library ? "cuLibraryLoadFromFile" : "cuModuleLoad");

    // kernel 的属性：寄存器数、它是由哪一代 SASS / PTX 来的
    int regs = 0, bin_ver = 0, ptx_ver = 0;
    CU_CHECK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f_vadd));
    CU_CHECK(cuFuncGetAttribute(&bin_ver, CU_FUNC_ATTRIBUTE_BINARY_VERSION, f_vadd));
    CU_CHECK(cuFuncGetAttribute(&ptx_ver, CU_FUNC_ATTRIBUTE_PTX_VERSION, f_vadd));
    std::printf("vadd: %d 个寄存器/线程, SASS 版本 sm_%d, PTX 版本 %d\n", regs, bin_ver, ptx_ver);

    // ---- 第 5 步：分配显存、拷数据（同步版本） ----
    std::vector<float> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; ++i) { ha[i] = (float)i; hb[i] = 2.0f * i; }
    CUdeviceptr da, db, dc;  // CUdeviceptr 就是一个 64 位整数：显存的虚拟地址
    const size_t bytes = (size_t)n * sizeof(float);
    CU_CHECK(cuMemAlloc(&da, bytes));
    CU_CHECK(cuMemAlloc(&db, bytes));
    CU_CHECK(cuMemAlloc(&dc, bytes));
    CU_CHECK(cuMemcpyHtoD(da, ha.data(), bytes));
    CU_CHECK(cuMemcpyHtoD(db, hb.data(), bytes));
    std::printf("显存地址 da = 0x%llx\n", (unsigned long long)da);

    // ---- 第 6 步：启动 kernel ----
    // <<<grid, block>>> 在这里变成 cuLaunchKernel 的 6 个数字；
    // 参数用「指针数组」传：数组的第 i 个元素 = 指向第 i 个参数的指针（第 5.2 节）
    int threads = 256, blocks = (n + threads - 1) / threads;
    void* args[] = {&da, &db, &dc, (void*)&n};
    t0 = now_us();
    CU_CHECK(cuLaunchKernel(f_vadd,
                            blocks, 1, 1,     // gridDim
                            threads, 1, 1,    // blockDim
                            0,                // 动态共享内存字节数
                            nullptr,          // stream：0 / nullptr = 默认 stream
                            args, nullptr));  // 参数；最后一个 extra 一般传 nullptr
    t1 = now_us();
    CU_CHECK(cuCtxSynchronize());  // 等 GPU 做完（kernel 启动是异步的）
    double t2 = now_us();
    std::printf("cuLaunchKernel 返回          %10.1f us   (首次启动)\n", t1 - t0);
    std::printf("启动到 kernel 做完           %10.1f us\n", t2 - t0);

    // 再启动一次，看第二次快多少（第一次可能包含懒加载 / 首次初始化）
    t0 = now_us();
    CU_CHECK(cuLaunchKernel(f_vadd, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    t1 = now_us();
    CU_CHECK(cuCtxSynchronize());
    t2 = now_us();
    std::printf("cuLaunchKernel 返回          %10.1f us   (第二次)\n", t1 - t0);
    std::printf("启动到 kernel 做完           %10.1f us   (第二次)\n", t2 - t0);

    // ---- 第 7 步：拷回、检查 ----
    CU_CHECK(cuMemcpyDtoH(hc.data(), dc, bytes));
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (hc[i] != 3.0f * i) { if (++bad <= 3) std::printf("错: c[%d]=%f\n", i, hc[i]); }
    std::printf("结果 %s\n", bad ? "WRONG" : "OK");

    // ---- 第 8 步：按相反顺序释放 ----
    CU_CHECK(cuMemFree(da));
    CU_CHECK(cuMemFree(db));
    CU_CHECK(cuMemFree(dc));
    if (mod) CU_CHECK(cuModuleUnload(mod));
    if (lib) CU_CHECK(cuLibraryUnload(lib));
    CU_CHECK(cuDevicePrimaryCtxRelease(dev));  // 引用计数 -1，到 0 时才真正销毁
    return bad ? 1 : 0;
}
