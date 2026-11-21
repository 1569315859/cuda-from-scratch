#include "common.cuh"
int main() {
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        printf("GPU %d: %s\n", i, p.name);
        printf("  compute capability : %d.%d\n", p.major, p.minor);
        printf("  SM count           : %d\n", p.multiProcessorCount);
        printf("  global memory      : %.0f MB\n", p.totalGlobalMem / 1048576.0);
        printf("  shared mem / block : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  max threads / block: %d\n", p.maxThreadsPerBlock);
        printf("  warp size          : %d\n", p.warpSize);
        printf("  memory bus width   : %d bit\n", p.memoryBusWidth);
    }
    return 0;
}
