// syscall_cost.c —— 预备 C：一次系统调用要多久？和一次普通函数调用比
// 编译：gcc -O2 syscall_cost.c -o syscall_cost
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
__attribute__((noinline)) static long plain_func(long x) { return x + 1; }   // noinline：不让编译器把它优化掉

int main(void) {
    const int N = 1000000;
    volatile long sink = 0;
    double t0 = now_ns();
    for (int i = 0; i < N; ++i) sink += plain_func(i);
    double t1 = now_ns();
    for (int i = 0; i < N; ++i) sink += syscall(SYS_getppid);   // 直接走系统调用，不经过 libc 的缓存
    double t2 = now_ns();
    printf("普通函数调用  %.2f ns/次\n", (t1 - t0) / N);
    printf("系统调用      %.2f ns/次（getppid，最简单的系统调用之一）\n", (t2 - t1) / N);
    return 0;
}
