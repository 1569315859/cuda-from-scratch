// hook.c —— 用 LD_PRELOAD 替换 mathx_add
// 编译：gcc -O2 -shared -fPIC hook.c -o libhook.so -ldl
// 运行：LD_PRELOAD=./libhook.so ./app
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

int mathx_add(int a, int b) {
    static int (*real)(int, int) = NULL;
    if (!real) real = (int (*)(int, int))dlsym(RTLD_NEXT, "mathx_add");   // RTLD_NEXT：按加载顺序，找「我后面」的那个同名函数
    printf("   [hook] 拦到 mathx_add(%d,%d)\n", a, b);
    return real(a, b) + 1000;   // 故意加 1000，一眼能看出走没走 hook
}
