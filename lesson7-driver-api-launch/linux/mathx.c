// mathx.c —— 编成 libmathx.so.1（动态库），扮演「驱动里的 libcuda.so.1」
// 编译：gcc -O2 -shared -fPIC -Wl,-soname,libmathx.so.1 mathx.c -o libmathx.so.1 && ln -sf libmathx.so.1 libmathx.so
#include <stdio.h>
#include <string.h>

int mathx_add(int a, int b) { return a + b; }
int mathx_scale(int x) { return x * 2; }          // 老版本 v1：还留着，给老程序用（注意：这里不 include mathx.h，否则会被宏改名）
long mathx_scale_v2(long x) { return x * 2; }      // 新版本 v2

void* mathx_get_proc(const char* name) {
    if (strcmp(name, "mathx_add") == 0) return (void*)mathx_add;
    if (strcmp(name, "mathx_scale") == 0) return (void*)mathx_scale_v2;   // 新语义：名字不带后缀，给你最新的实现
    return NULL;
}

// 构造函数：库一被加载就执行，比 main 还早 —— 和 nvcc 生成的 __cudaRegisterAll 同一个机制
__attribute__((constructor)) static void on_load(void) { printf("[libmathx] 构造函数：库被加载了\n"); }
