// mathx.h —— 一个「迷你 libcuda」的头文件：模仿 cuda.h 的写法
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int mathx_add(int a, int b);
// 版本化：老的 mathx_scale 返回 int，新的 _v2 返回 long（像 cuMemAlloc 从 32 位改成 64 位大小）
// 头文件用宏让新代码自动用新名字 —— 和 cuda.h 里 #define cuMemAlloc cuMemAlloc_v2 一模一样
long mathx_scale_v2(long x);
#define mathx_scale mathx_scale_v2
// 模仿 cuGetProcAddress：按名字把函数地址交出去
void* mathx_get_proc(const char* name);
#ifdef __cplusplus
}
#endif
