// app.c —— 用三种方式调用 libmathx 里的 mathx_add：
//   ① 直接调用（链接时 -lmathx，由动态链接器解析符号）  ← 像程序直接调用 cuLaunchKernel
//   ② dlopen + dlsym(句柄, "mathx_add")                 ← 像 runtime 用 dlsym 拿 cuGetProcAddress
//   ③ mathx_get_proc("mathx_add")                       ← 像 runtime 用 cuGetProcAddress 拿其他函数
// 编译：gcc -O2 app.c -o app -L. -lmathx -ldl -Wl,-rpath,'$ORIGIN'
#include <dlfcn.h>
#include <stdio.h>
#include "mathx.h"

__attribute__((constructor)) static void before_main(void) { printf("[app] 构造函数：main 之前\n"); }

int main(void) {
    printf("[app] main 开始\n");
    printf("① 直接调用          mathx_add(1,2) = %d\n", mathx_add(1, 2));

    void* h = dlopen("libmathx.so.1", RTLD_NOW | RTLD_NOLOAD);   // 已经加载了，只拿句柄
    int (*f2)(int, int) = (int (*)(int, int))dlsym(h, "mathx_add");
    printf("② dlsym(句柄)       mathx_add(1,2) = %d\n", f2(1, 2));

    int (*f3)(int, int) = (int (*)(int, int))mathx_get_proc("mathx_add");
    printf("③ get_proc          mathx_add(1,2) = %d\n", f3(1, 2));

    printf("   mathx_scale(5) = %ld（源码写的是 mathx_scale，实际调用的是 mathx_scale_v2）\n", mathx_scale(5));
    printf("   地址：① %p  ② %p  ③ %p\n", (void*)mathx_add, (void*)f2, (void*)f3);
    return 0;
}
