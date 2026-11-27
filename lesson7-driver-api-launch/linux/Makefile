# 第 7 课「Linux 预备」的小程序（不需要 GPU，WSL 里直接跑）
#   make            全部编译
#   make tls        预备 A：线程局部变量
#   make vm         预备 B：虚拟地址空间、mmap、缺页
#   make sys        预备 C：系统调用耗时 + strace
#   make lib        预备 D：动态库、nm、ldd、readelf、构造函数
#   make hook       预备 E：LD_PRELOAD 能拦到哪种调用
#   make hook2      预备 E：库用 -Bsymbolic 编译后，再看一次（结果和 hook 不一样！）
CC = gcc
CXX = g++
all: tls_demo vm_demo syscall_cost libmathx.so.1 app libhook.so

tls_demo: tls_demo.cpp
	$(CXX) -O2 -std=c++17 -pthread $< -o $@
vm_demo: vm_demo.c
	$(CC) -O0 $< -o $@
syscall_cost: syscall_cost.c
	$(CC) -O2 $< -o $@
libmathx.so.1: mathx.c
	$(CC) -O2 -shared -fPIC -Wl,-soname,libmathx.so.1 $< -o $@
	ln -sf libmathx.so.1 libmathx.so
app: app.c mathx.h libmathx.so.1
	$(CC) -O2 $< -o $@ -L. -lmathx -ldl -Wl,-rpath,'$$ORIGIN'
libmathx_sym/libmathx.so.1: mathx.c
	mkdir -p libmathx_sym
	$(CC) -O2 -shared -fPIC -Wl,-Bsymbolic -Wl,-soname,libmathx.so.1 $< -o $@
libhook.so: hook.c
	$(CC) -O2 -shared -fPIC $< -o $@ -ldl

tls: tls_demo
	./tls_demo
vm: vm_demo
	./vm_demo
sys: syscall_cost
	./syscall_cost
	strace -c -f ls > /dev/null
lib: app
	@echo "===== ldd app：app 依赖哪些动态库、各在哪 ====="; ldd ./app
	@echo "===== nm -D app：app 要从外面导入（U）哪些函数 ====="; nm -D ./app | grep -E " U (mathx|dl)"
	@echo "===== nm -D libmathx.so.1：库导出（T）哪些函数 ====="; nm -D libmathx.so.1 | grep -E " T "
	@echo "===== readelf -d：库的 soname ====="; readelf -d libmathx.so.1 | grep SONAME
	@echo "===== readelf -S app：ELF 里的几个段 ====="; readelf -S -W ./app | grep -E " \.(text|data|bss|rodata|plt|got|dynsym|init_array) "
	@echo "===== 运行 ====="; ./app
hook: app libhook.so
	@echo "===== 不加 hook ====="; ./app
	@echo "===== LD_PRELOAD=./libhook.so ====="; LD_PRELOAD=./libhook.so ./app
clean:
	rm -rf tls_demo vm_demo syscall_cost libmathx.so* libmathx_sym app libhook.so
hook2: app libhook.so libmathx_sym/libmathx.so.1
	@echo "===== 换成 -Bsymbolic 版的库 + LD_PRELOAD ====="; LD_LIBRARY_PATH=./libmathx_sym LD_PRELOAD=./libhook.so ./app
.PHONY: all tls vm sys lib hook hook2 clean
