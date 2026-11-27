// tls_demo.cpp —— 预备 A：普通全局变量 vs thread_local（线程局部）变量
// 编译：g++ -O2 -std=c++17 -pthread tls_demo.cpp -o tls_demo
#include <atomic>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <sys/syscall.h>

std::atomic<int> g_shared{0};     // 普通全局变量：整个进程只有一份，所有线程看到的是同一个（用 atomic，两个线程同时加也不会丢）
thread_local int t_mine = 0;      // 线程局部变量：每个线程各有一份，互不影响

void work(const char* name, int times) {
    for (int i = 0; i < times; ++i) { ++g_shared; ++t_mine; }
    usleep(100000);  // 睡 0.1 秒，保证两个线程同时活着（否则 B 可能复用 A 留下的那块内存，地址会一样）
    std::printf("%s: 进程号 pid=%d 线程号 tid=%ld  &g_shared=%p  &t_mine=%p  t_mine=%d\n",
                name, getpid(), (long)syscall(SYS_gettid), (void*)&g_shared, (void*)&t_mine, t_mine);
}

int main() {
    std::thread a(work, "线程A", 3);
    std::thread b(work, "线程B", 5);
    a.join();
    b.join();
    work("主线程", 1);
    std::printf("g_shared = %d（3+5+1，大家共用一份）\n", g_shared.load());
    return 0;
}
