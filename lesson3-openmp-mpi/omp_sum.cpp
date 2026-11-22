// omp_sum.cpp —— 用 OpenMP 给 N 个数求和：正确写法、错误写法、伪共享演示
// 编译：g++ -O2 -std=c++17 -fopenmp omp_sum.cpp -o omp_sum
// 运行：OMP_NUM_THREADS=4 ./omp_sum [N]       （N 默认 2^25 ≈ 3355 万）
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 一个「独占一条缓存行」的计数器：alignas(64) 让每个元素从 64 字节边界开始，
// 并把大小撑到 64 字节，这样不同线程的计数器不会挤在同一条缓存行里
struct alignas(64) PaddedDouble { double v; };

int main(int argc, char** argv) {
    const long n = (argc > 1) ? atol(argv[1]) : (1L << 25);
    const int T = omp_get_max_threads();          // 本次运行最多用几个线程（受 OMP_NUM_THREADS 控制）
    std::vector<double> a(n);
    for (long i = 0; i < n; ++i) a[i] = (i % 7) * 0.5;   // 简单、可预测的数据
    const double* p = a.data();
    printf("N = %ld, threads = %d\n", n, T);

    // 0) 预热：先完整读一遍数组，让 CPU 升频、TLB/页表就绪，避免第一个被测版本吃亏
    { volatile double w = 0.0; for (long i = 0; i < n; ++i) w = w + p[i]; (void)w; }
    #pragma omp parallel
    { volatile int dummy = omp_get_thread_num(); (void)dummy; }   // 预先创建线程池

    // 1) 串行基准
    double t = omp_get_wtime();
    double ref = 0.0;
    for (long i = 0; i < n; ++i) ref += p[i];
    double t_serial = omp_get_wtime() - t;
    printf("%-26s time %8.3f ms  sum %.1f\n", "serial", t_serial * 1e3, ref);

    // 2) 错误写法：所有线程同时改 shared 变量 sum —— 竞争条件（race condition）
    t = omp_get_wtime();
    double sum_race = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < n; ++i) sum_race += p[i];      // 故意写错！
    double t_race = omp_get_wtime() - t;
    printf("%-26s time %8.3f ms  sum %.1f  %s\n", "race (WRONG on purpose)", t_race * 1e3,
           sum_race, sum_race == ref ? "same (lucky)" : "DIFFERENT -> race!");

    // 3) atomic：正确但每次加法都要排队，很慢
    t = omp_get_wtime();
    double sum_atomic = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < n; ++i) {
        #pragma omp atomic
        sum_atomic += p[i];
    }
    double t_atomic = omp_get_wtime() - t;
    printf("%-26s time %8.3f ms  sum %.1f\n", "atomic", t_atomic * 1e3, sum_atomic);

    // 4) 每个线程一个格子，但格子挨在一起 —— 伪共享（false sharing）
    t = omp_get_wtime();
    std::vector<double> part(T, 0.0);
    double* pp = part.data();
    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n; ++i) pp[id] += p[i];    // 每次都写内存里的 pp[id]
    }
    double sum_fs = 0.0;
    for (int k = 0; k < T; ++k) sum_fs += part[k];
    double t_fs = omp_get_wtime() - t;
    printf("%-26s time %8.3f ms  sum %.1f\n", "partial[] (false sharing)", t_fs * 1e3, sum_fs);

    // 5) 同样的思路，但每个格子独占一条缓存行 —— 没有伪共享
    t = omp_get_wtime();
    std::vector<PaddedDouble> part2(T, PaddedDouble{0.0});
    PaddedDouble* pp2 = part2.data();
    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n; ++i) pp2[id].v += p[i];
    }
    double sum_pad = 0.0;
    for (int k = 0; k < T; ++k) sum_pad += part2[k].v;
    double t_pad = omp_get_wtime() - t;
    printf("%-26s time %8.3f ms  sum %.1f\n", "partial[] (padded)", t_pad * 1e3, sum_pad);

    // 6) 标准写法：reduction —— 每个线程私有一份 sum，最后自动合并
    t = omp_get_wtime();
    double sum_red = 0.0;
    #pragma omp parallel for reduction(+:sum_red)
    for (long i = 0; i < n; ++i) sum_red += p[i];
    double t_red = omp_get_wtime() - t;
    printf("%-26s time %8.3f ms  sum %.1f\n", "reduction", t_red * 1e3, sum_red);

    printf("speedup of reduction vs serial: %.2fx\n", t_serial / t_red);
    // 浮点加法顺序不同，结果可能在最后几位有差别；这里的数据都是 0.5 的倍数，应当完全一致
    bool ok = (sum_atomic == ref) && (sum_fs == ref) && (sum_pad == ref) && (sum_red == ref);
    printf("correct versions match serial: %s\n", ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}
