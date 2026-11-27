// vm_demo.c —— 预备 B：虚拟地址空间长什么样、malloc 和 mmap、「分配」不等于「真的占用物理内存」
// 编译：gcc -O0 vm_demo.c -o vm_demo
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int g_init = 42;        // 已初始化的全局变量 → .data 段
int g_zero;             // 未初始化的全局变量 → .bss 段

// 读 /proc/self/statm 的第 2 个数：常驻内存（RSS），单位是页
static long rss_kb(void) {
    long size, resident;
    FILE* f = fopen("/proc/self/statm", "r");
    if (fscanf(f, "%ld %ld", &size, &resident) != 2) resident = -1;
    fclose(f);
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

int main(void) {
    int local = 1;                              // 局部变量 → 栈
    int* small = malloc(16);                    // 小块 malloc → 堆（brk 扩出来的那块）
    printf("页大小               %ld 字节\n", sysconf(_SC_PAGESIZE));
    printf("代码 main            %p\n", (void*)main);
    printf("全局 g_init (.data)  %p\n", (void*)&g_init);
    printf("全局 g_zero (.bss)   %p\n", (void*)&g_zero);
    printf("堆   small           %p\n", (void*)small);
    printf("栈   local           %p\n", (void*)&local);

    // 申请 256 MB 匿名内存。mmap 只是「在地址空间里划一块」，还没有分配物理内存
    size_t sz = 256u << 20;
    long r0 = rss_kb();
    char* big = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    long r1 = rss_kb();
    printf("mmap 256 MB          %p   RSS %ld KB → %ld KB（几乎不变）\n", (void*)big, r0, r1);
    memset(big, 1, sz / 2);                    // 真正写一半：每碰到一个新页就触发一次「缺页」，内核这时才给物理页
    long r2 = rss_kb();
    printf("写了前 128 MB 之后   RSS %ld KB（多了约 %ld MB）\n", r2, (r2 - r1) / 1024);

    // 打印地址空间里的几段（/proc/self/maps 每一行就是一段）
    printf("\n/proc/self/maps 的一部分：\n");
    FILE* f = fopen("/proc/self/maps", "r");
    char line[512];
    while (fgets(line, sizeof line, f))
        if (strstr(line, "vm_demo") || strstr(line, "[heap]") || strstr(line, "[stack]") || strstr(line, "libc"))
            fputs(line, stdout);
    fclose(f);
    munmap(big, sz);
    free(small);
    return 0;
}
