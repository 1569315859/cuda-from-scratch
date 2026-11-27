// vadd_kernel.cu —— 只有 kernel、没有 main。它不直接编成可执行文件，而是编成三种「GPU 代码文件」：
//   vadd.ptx     PTX：虚拟指令集的文本，加载时由驱动 JIT 编译成本机 SASS（第 3 章）
//   vadd.cubin   cubin：已经是 sm_86 的 SASS 机器码，加载时不用编译
//   vadd.fatbin  fatbin：一个「胖」容器，里面可以同时装多份 cubin 和 PTX，驱动挑最合适的一份
// 然后由 drv_vadd（纯 driver API）在运行时用 cuModuleLoad / cuLibraryLoad 把它们加载进来。
//
// extern "C"：关掉 C++ 的名字修饰（name mangling）。不加的话函数名会变成 _Z4vaddPKfS0_Pfi，
// 用 cuModuleGetFunction(&f, mod, "vadd") 就找不到它了。

extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 第二个 kernel：什么也不干，用来量「启动本身」要多久
extern "C" __global__ void empty_kernel() {}

// 第三个 kernel：故意越界写，用来演示 sticky（粘住的）错误（第 6 章）
extern "C" __global__ void bad_write(float* p, long long offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[offset] = 1.0f;
}
