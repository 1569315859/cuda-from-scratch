## Lesson 7 — Driver API, contexts, modules and the kernel launch path (Runtime track, part 1)

RTX 2050 Laptop (GA107, sm_86, 16 SMs, 4 GB), WSL2 Ubuntu, CUDA 13.5, driver 620.73. All timings are host-side (`std::chrono`) unless noted; T5 uses cudaEvent.

| Program | What it shows |
|---|---|
| `drv_vadd` | Vector add with the driver API only (`cuInit` → primary ctx → `cuModuleLoad`/`cuLibraryLoadFromFile` → `cuLaunchKernel`), loading `vadd.cubin` / `vadd.ptx` / `vadd.fatbin` / PTX-only fatbin (JIT) |
| `ctx_explore` | Runtime ↔ driver context interop: primary context, per-thread current context, cost and memory of `cuCtxCreate`, which context owns `cudaMalloc` memory |
| `launch_overhead` | First-launch cost (LAZY vs EAGER), async submit cost and queue back-pressure, launch+sync round trip, `<<<>>>` vs `cudaLaunchKernel` vs `cuLaunchKernel`, empty-grid dispatch cost |
| `errors` | Non-sticky (`cudaErrorInvalidConfiguration`) vs sticky (`cudaErrorIllegalAddress`) errors, `cudaDeviceReset` recovery |
| `proc_addr` | `dlsym` vs `cuGetProcAddress` vs `cudaGetDriverEntryPointByVersion` for versioned symbols (`_v2`, `_v4`, `_ptsz`) |
| `libhook_preview.so` | LD_PRELOAD hook of `cuLaunchKernel` (lesson 10 preview) |

### Verified in the sandbox (no GPU; nvcc 13.5)

- `ptxas -v`: `vadd` 12 regs / 380 B cmem[0]; `empty_kernel` 4 regs / 352 B cmem[0] (difference 28 B = 3 pointers + 1 int).
- `nvcc --keep`: `<<<2,64>>>(7)` becomes `__cudaPushCallConfiguration(2, 64)` + a host stub that collects argument addresses, `__cudaPopCallConfiguration`, `__cudaGetKernel`, `__cudaLaunchKernel(cudaKernel_t, ...)`. Kernels are registered before `main` via `__cudaRegisterFatBinary` / `__cudaRegisterFunction`.
- Statically linked cudart imports no driver functions: `nm -D errors | grep " U cu"` shows only the two the program calls itself (`cuCtxSynchronize`, `cuGetErrorName`), while `strings` finds 527 `cu*` names plus `libcuda.so.1` and `cuGetProcAddress_v2` — the runtime resolves the driver at run time.

### Results (fill in)

| Item | Measured |
|---|---|
| `cuInit` / primary ctx / `cudaFree(0)` (LAZY, EAGER) | TBD |
| `cuCtxCreate` time / device memory per context | TBD |
| Load cubin / ptx / fatbin / library; PTX-only no-cache / cache miss / cache hit | TBD |
| First launch (LAZY / EAGER), second launch | TBD |
| Async submit per launch: first 500 / last 500 / median / p99 | TBD |
| GPU throughput per empty kernel; launch+sync round trip | TBD |
| `<<<>>>` / `cudaLaunchKernel` / `cuLaunchKernel` | TBD |
| Empty kernel, grid 1 / 16 / 256 / 4096 / 65536 | TBD |
| Hook count: `drv_vadd` (pred. 2), `launch_overhead` (pred. 20000, only the direct driver calls) | TBD |
| `ioctl` count for `drv_vadd`; devices opened (`/dev/dxg`) | TBD |
