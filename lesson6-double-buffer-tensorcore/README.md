## Lesson 6 — Double buffering, cp.async, warp tiling and Tensor Cores (GEMM wrap-up)

RTX 2050 Laptop (GA107, sm_86, 16 SMs, 4 GB), WSL2 Ubuntu, CUDA 13.5, driver 620.73. Row-major, N × N.
Timing: cudaEvent, 0.5 s GPU wake-up, 1 warm-up + 10 timed runs, average. Correctness vs CPU i-k-j (N ≤ 2048) or cuBLAS FP32 (N > 2048): FP32 kernels max_rel_err < 1e-3, TF32/FP16 paths < 1e-2. All hand-written kernels require N to be a multiple of 128 (`wmma_naive`: 32). FP16 inputs are converted from FP32 once, before timing (conversion not timed).

| Kernel | Idea | Threads/block | Smem | Regs/thread (nvcc 13.5 `-Xptxas -v`; measured 2026-11-27: identical) | Theor. occupancy (calc.; program printout identical) |
|---|---|---|---|---|---|
| `sgemm_2d_vec` | lesson 5 baseline: 128×128 block tile, 8×8 per thread, float4, transposed A | 256 | 8 KB | 117 (measured by me: 117), 0 spill | 33.3% |
| `sgemm_db_reg` | + double buffering via register prefetch, 2 smem buffers, 1 `__syncthreads` per k-step; `__launch_bounds__(256, 2)` | 256 | 16 KB | 128 (129 without the min-blocks hint → 1 block/SM), 0 spill | 33.3% |
| `sgemm_db_cpasync` | + `cp.async` (LDGSTS) 2-stage pipeline, global → shared without registers | 256 | 16 KB | 117, 0 spill | 33.3% |
| `sgemm_warptile` | + warp tiling: block 128×128 → warp 64×32 → thread 2×2 pieces of 4×4 | 256 | 16 KB | 119, 0 spill | 33.3% |
| `sgemm_wmma_naive` | Tensor Core intro: WMMA 16×16×16, one warp = one 16×16 C tile, loads straight from global | 128 | 0 | 40, 0 spill | 100% |
| `sgemm_wmma_fp16` | WMMA + 128×128 block tile + 64×32 warp tile + cp.async double buffering, padded smem (ldmatrix conflict-free) | 256 | 37,888 B | 122, 0 spill | 33.3% |
| `cublas` / `cublas_tf32` / `cublas_fp16` | `cublasSgemm` / `cublasGemmEx` (TF32 compute) / `cublasGemmEx` (FP16 in, FP32 accumulate) | — | — | — | — |

Reference SASS per kernel (static counts; `make sass` on my machine 2026-11-27 matched every count): cp.async kernels show `LDGSTS.E` ×8 + `LDGSTS.E.BYPASS.128` ×2, `LDGDEPBAR`, `DEPBAR.LE SB0,0x0`; `wmma_fp16` shows 32 × `HMMA.16816.F32`, 8 × `LDSM.16.M88.4`, 4 × `LDSM.16.MT88.4` per k-step.

### Performance (measured 2026-11-27, N=1024 and N=2048; 1 warm-up + 10 timed runs, average)

| Kernel | N=1024 ms | N=1024 GFLOPS | N=2048 GFLOPS | vs 2d_vec | % of cuBLAS FP32 | max_rel_err |
|---|---|---|---|---|---|---|
| 2d_vec | 0.727 | 2953.0 | 2866.7 | 1.00× | 69.8% | 3.54e-07 |
| db_reg | 0.631 | 3401.2 | 3289.6 | 1.15× | 80.4% | 3.54e-07 |
| db_cpasync | 0.832 | 2582.2 | 2529.7 | 0.87× | 61.1% | 3.54e-07 |
| warptile | 0.714 | 3006.9 | 2934.3 | 1.02× | 71.1% | 3.54e-07 |
| wmma_naive (FP16) | 1.080 | 1989.3 | 744.4 | 0.67× | 47.0% | 6.35e-05 |
| wmma_fp16 (FP16) | 0.213 | 10105.2 | 11043.4 | 3.42× | 238.9% | 6.35e-05 |
| cuBLAS FP32 | 0.508 | 4229.6 | 3830.4 | 1.43× | 100% | 2.27e-06 |
| cuBLAS TF32 | 0.356 | 6026.2 | 5147.7 | 2.04× | 142.5% | 6.49e-05 |
| cuBLAS FP16 | 0.266 | 8085.4 | 11916.1 | 2.74× | 191.2% | 5.77e-05 |

Theoretical peaks for this GPU (derived from the GA102 whitepaper per-SM ratios, 16 SMs × 1.245–1.477 GHz): FP32 FFMA 5.1–6.0 TFLOPS, TF32 Tensor 5.1–6.0, FP16 Tensor with FP32 accumulate 10.2–12.1, FP16 accumulate 20.4–24.2.


### Sweep (GFLOPS, `make sweep`, 2026-11-27)

| Kernel | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|
| 2d_vec | 2521.1 | 2922.0 | 2843.7 | 2704.0 |
| db_reg | 3558.6 | 3394.5 | 3197.0 | 2493.1 |
| db_cpasync | 2581.3 | 2625.4 | 2502.3 | 2395.3 |
| warptile | 2985.7 | 3100.9 | 2877.8 | 2271.0 |
| wmma_naive | 2986.2 | 1921.0 | 741.1 | 595.6 |
| wmma_fp16 | 7934.0 | 9804.6 | 10986.3 | 9698.5 |
| cuBLAS FP32 / TF32 / FP16 | 2963.4 / 5400.9 / 8900.4 | 4228.4 / 6116.7 / 8074.7 | 3820.9 / 5069.5 / 12035.3 | 2579.6 / 5864.4 / 10841.9 |

N=4096: all FP32 kernels and cuBLAS FP32 drop together (likely laptop clock throttling, unconfirmed); rankings at 4096 not treated as conclusive.

### Nsight Compute evidence (N=1024, measured 2026-11-27)

Also: shared-load bank conflicts 4,194,304 (2d_vec/db_reg/db_cpasync) → 0 (warptile), 4,167 (wmma_fp16); MIO throttle 0.57 / 0.71 / 1.92 / 1.12 / 40.78 / 2.79; DRAM read 17.7 / 18.5 / 18.3 / 18.5 / 88.1 / 9.1 MB.


| Metric | 2d_vec | db_reg | db_cpasync | warptile | wmma_naive | wmma_fp16 |
|---|---|---|---|---|---|---|
| shared ld instructions (predicted 2,097,152 for FP32 kernels) | 2,097,152 | 2,097,152 | 2,292,224 | 2,292,224 | 0 | 47,616 (LDSM not counted) |
| shared ld wavefronts (predicted ≈5.24M → ≈2.10M with warp tiling) | 10,486,096 | 10,485,958 | 10,485,873 | 6,291,555 | 0 | 790,954 |
| LDGSTS instructions (predicted 327,680 / 65,536) | 0 | 0 | 327,680 | 327,680 | 0 | 65,536 |
| FFMA thread inst. (= N³) / HMMA warp inst. (= N³/2048 = 524,288) | 1,073,741,824 | same | same | same | 524,288 | 524,288 |
| stall long_scoreboard (per issue) | 0.95 | 0.03 | 0.09 | 0.06 | 51.75 | 1.31 |
| stall barrier (per issue) | 0.72 | 0.61 | 1.00 | 0.61 | 0 | 10.25 |
| FMA / tensor pipe utilization % | 52.1 | 60.5 | 45.5 | 53.1 | 9.3 (tensor) | 43.1 (tensor) |
| achieved occupancy % | 31.1 | 30.8 | 30.7 | 30.8 | 107 (as reported) | 31.1 |

### Correctness tooling

- `make racecheck`: normal build 0 hazards; `-DSKIP_DB_SYNC` (db_reg without its only barrier) → 5 hazard sites (1 error, 4 warnings; write lines 287 / 221–224 vs read line 282), result WRONG (1.00e-01).
- `make cpwait-bug`: `-DSKIP_CP_WAIT` (no `cp.async.wait_group`) → plain run WRONG for all three (1.3e-02 / 1.6e-02 / 3.0e-02) at unchanged speed; racecheck catches it (28 hazard sites, cp_async writes vs shared reads) although results are correct under the sanitizer; synccheck reports 0 errors. `make memcheck`: 0 errors.
- `make emu`: CPU-thread emulation of all six kernels (no GPU needed) — all OK at N=128/256; the no-wait and no-sync builds produce WRONG results.

**Finding (all measured 2026-11-27):** Register-prefetch double buffering (1 barrier per k-step) cut long-scoreboard stalls 0.95 → 0.03 and gave +15.2% over 2d_vec (3401.2 vs 2953.0 GFLOPS at N=1024, +14.8% at 2048) — best FP32 kernel, 80.4% of cuBLAS FP32. cp.async was slower (2582.2, 0.87×): transposing A forces 4-byte LDGSTS, 327,680 vs 131,072 global load/copy instructions (ncu: MIO throttle 0.57 → 1.92). Warp tiling on the same cp.async base, with identical instruction counts, gained +16.4% (3006.9); bank conflicts 4.19M → 0, shared-load wavefronts −40% (10.49M → 6.29M; not the predicted 2.10M — LDS.128 is processed per half-warp, see lesson 5). Tensor Cores (FP16 in, FP32 accumulate): wmma_naive 1989.3 → 744.4 GFLOPS from N=1024 to 2048 (8 FLOP/B, likely DRAM-bound once A/B exceed L2 — inferred), wmma_fp16 10105.2 (beats cuBLAS FP16 8085.4 at 1024) and 11043.4 = 92.7% of cuBLAS FP16 at 2048. cuBLAS TF32 is 1.42× cuBLAS FP32 despite equal theoretical peaks: SGEMM reaches ~70% of FP32 peak; measured TF32/FP16 rates at or above my 1.477 GHz peak estimates suggest the GPU boosts higher (clock check pending). N=2048 runs showed >10% avg-vs-min spread (laptop clock variance). Next: lesson 7 — driver API, contexts, modules, and the kernel launch path.

### cuBLAS kernels (`make ncu-cublas`, N=1024)

FP32 `ampere_sgemm_128x64_nn` (grid 8×16×2, 122 regs, FMA 62.0%, no HMMA); TF32 `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4` (224 regs, 1,048,576 HMMA, tensor 45.7%); FP16 `cutlass_80_tensorop_s16816gemm_f16_256x128_32x3_nn_align8` (grid 32×1×2, 218 regs, 524,288 HMMA, tensor 32.9%).

### Clock note

ncu locks SM clock at 1.24 GHz; cycles ÷ benchmark time implies ~1.4–1.5 GHz during `make run`. At that clock cuBLAS TF32 (6116.7 GFLOPS) would be ~100% of the GA102-ratio TF32 estimate while ncu shows the tensor pipe only 45.7% busy — the per-SM Tensor throughput estimate for GA107 is likely too low (unresolved; needs nvidia-smi clocks + an HMMA microbenchmark).
