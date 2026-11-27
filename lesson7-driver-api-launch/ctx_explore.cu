// ctx_explore.cu —— 实验 2：runtime 和 driver 是怎么共用 context 的（第 2 章）
//
// 依次回答 6 个问题，每个问题打印一行「预测 / 实际」：
//   Q1 进程刚启动时有没有 context？            cuCtxGetCurrent
//   Q2 runtime 的第一次调用做了什么、花多久？  cudaFree(0)
//   Q3 runtime 用的是不是 primary context？    和 cuDevicePrimaryCtxRetain 比指针
//   Q4 一个 context 吃多少显存？               cuCtxCreate 前后 cuMemGetInfo
//   Q5 当前 context 是「每个线程一份」吗？     开一个新线程看 cuCtxGetCurrent
//   Q6 别的 context 是当前时，cudaMalloc 的内存归谁？  cuPointerGetAttribute
//
// 编译：nvcc -O2 -arch=sm_86 ctx_explore.cu -o ctx_explore -lcuda

#include <thread>
#include "common7.h"

__global__ void touch(int* p) { *p = 42; }

static unsigned long long ctx_id(CUcontext c) {
    unsigned long long id = 0;
    if (c) CU_CHECK(cuCtxGetId(c, &id));  // CUDA 12.0 起每个 context 有一个进程内唯一的 ID
    return id;
}

int main() {
    double t0, t1;
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));

    // ---------- Q1 ----------
    CUcontext cur = nullptr;
    CU_CHECK(cuCtxGetCurrent(&cur));
    unsigned flags = 0; int active = 0;
    CU_CHECK(cuDevicePrimaryCtxGetState(dev, &flags, &active));
    std::printf("Q1 启动时 current context = %p，primary context active = %d（预测：nullptr, 0）\n", (void*)cur, active);

    // ---------- Q2 ----------
    // cudaFree(0) 是老习惯：释放空指针什么也不做，但会逼 runtime 完成初始化 + 建好 primary context
    t0 = now_us();
    CUDA_CHECK(cudaFree(0));
    t1 = now_us();
    CU_CHECK(cuCtxGetCurrent(&cur));
    CU_CHECK(cuDevicePrimaryCtxGetState(dev, &flags, &active));
    std::printf("Q2 cudaFree(0) 用时 %.1f us；之后 current = %p (id %llu)，primary active = %d\n",
                t1 - t0, (void*)cur, ctx_id(cur), active);

    // ---------- Q3 ----------
    CUcontext primary;
    CU_CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
    std::printf("Q3 cuDevicePrimaryCtxRetain 拿到 %p (id %llu) —— 和 runtime 的 %s\n",
                (void*)primary, ctx_id(primary), primary == cur ? "是同一个" : "不是同一个！");

    // ---------- Q4 ----------
    size_t free0, total;
    CU_CHECK(cuMemGetInfo(&free0, &total));
    CUcontext c2;
    t0 = now_us();
    CU_CHECK(cuCtxCreate(&c2, nullptr, 0, dev));  // CUDA 13 的签名：多一个 CUctxCreateParams*，传 nullptr 即可
    t1 = now_us();
    size_t free1;
    CU_CHECK(cuMemGetInfo(&free1, &total));
    std::printf("Q4 cuCtxCreate 用时 %.1f us；显存空闲 %.1f MB → %.1f MB，这个 context 占了约 %.1f MB（总共 %.0f MB）\n",
                t1 - t0, free0 / 1048576.0, free1 / 1048576.0, ((double)free0 - (double)free1) / 1048576.0,
                total / 1048576.0);
    CU_CHECK(cuCtxGetCurrent(&cur));
    std::printf("   cuCtxCreate 之后 current = id %llu（预测：它会被压到栈顶，变成新建的 c2 = id %llu）\n",
                ctx_id(cur), ctx_id(c2));

    // ---------- Q6（先做，因为现在 c2 正好是 current）----------
    int* p_rt = nullptr;
    CUDA_CHECK(cudaMalloc(&p_rt, sizeof(int)));
    CUcontext owner = nullptr;
    CU_CHECK(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)p_rt));
    std::printf("Q6 c2 是 current 时 cudaMalloc 的内存属于 id %llu（primary = %llu，c2 = %llu）\n",
                ctx_id(owner), ctx_id(primary), ctx_id(c2));
    touch<<<1, 1>>>(p_rt);  // runtime 的 <<<>>> 也会在 c2 里启动
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(p_rt));

    // 销毁 c2：它是当前的，所以会被弹出，栈顶回到 primary
    CU_CHECK(cuCtxDestroy(c2));
    CU_CHECK(cuCtxGetCurrent(&cur));
    std::printf("   cuCtxDestroy(c2) 之后 current = id %llu（预测：回到 primary %llu）\n", ctx_id(cur), ctx_id(primary));

    // ---------- Q5 ----------
    std::thread th([&] {
        CUcontext in_thread = nullptr;
        CU_CHECK(cuCtxGetCurrent(&in_thread));
        std::printf("Q5 新线程里 current = %p（预测：nullptr —— current context 是线程局部的）\n", (void*)in_thread);
        // runtime 的调用会自动把 primary context 设成这个线程的 current
        CUDA_CHECK(cudaFree(0));
        CU_CHECK(cuCtxGetCurrent(&in_thread));
        std::printf("   新线程调用 cudaFree(0) 之后 current = id %llu（预测：primary %llu）\n",
                    ctx_id(in_thread), ctx_id(primary));
    });
    th.join();

    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
    std::printf("完成\n");
    return 0;
}
