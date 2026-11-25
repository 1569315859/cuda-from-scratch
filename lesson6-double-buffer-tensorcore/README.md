## Lesson 6 — Double buffering, cp.async, warp tiling and Tensor Cores (GEMM wrap-up)

RTX 2050 Laptop (GA107, sm_86, 16 SMs, 4 GB), WSL2 Ubuntu, CUDA 13.5, driver 620.73. Row-major, N × N.
Timing: cudaEvent, 0.5 s GPU wake-up, 1 warm-up + 10 timed runs, average. Correctness vs CPU i-k-j (N ≤ 2048) or cuBLAS FP32 (N > 2048): FP32 kernels max_rel_err < 1e-3, TF32/FP16 paths < 1e-2. All hand-written kernels require N to be a multiple of 128 (`wmma_naive`: 32). FP16 inputs are converted from FP32 once, before timing (conversion not timed).

| Kernel | Idea | Threads/block | Smem | Regs/thread (nvcc 13.5 `-Xptxas -v`, reference) | Theor. occupancy (calc.) |
|---|---|---|---|---|---|
| `sgemm_2d_vec` | lesson 5 baseline: 128×128 block tile, 8×8 per thread, float4, transposed A | 256 | 8 KB | 117 (measured by me: 117), 0 spill | 33.3% |
| `sgemm_db_reg` | + double buffering via register prefetch, 2 smem buffers, 1 `__syncthreads` per k-step; `__launch_bounds__(256, 2)` | 256 | 16 KB | 128 (129 without the min-blocks hint → 1 block/SM), 0 spill | 33.3% |
| `sgemm_db_cpasync` | + `cp.async` (LDGSTS) 2-stage pipeline, global → shared without registers | 256 | 16 KB | 117, 0 spill | 33.3% |
| `sgemm_warptile` | + warp tiling: block 128×128 → warp 64×32 → thread 2×2 pieces of 4×4 | 256 | 16 KB | 119, 0 spill | 33.3% |
| `sgemm_wmma_naive` | Tensor Core intro: WMMA 16×16×16, one warp = one 16×16 C tile, loads straight from global | 128 | 0 | 40, 0 spill | 100% |
| `sgemm_wmma_fp16` | WMMA + 128×128 block tile + 64×32 warp tile + cp.async double buffering, padded smem (ldmatrix conflict-free) | 256 | 37,888 B | 122, 0 spill | 33.3% |
| `cublas` / `cublas_tf32` / `cublas_fp16` | `cublasSgemm` / `cublasGemmEx` (TF32 compute) / `cublasGemmEx` (FP16 in, FP32 accumulate) | — | — | — | — |

Reference SASS per kernel (static counts): cp.async kernels show `LDGSTS.E` ×8 + `LDGSTS.E.BYPASS.128` ×2, `LDGDEPBAR`, `DEPBAR.LE SB0,0x0`; `wmma_fp16` shows 32 × `HMMA.16816.F32`, 8 × `LDSM.16.M88.4`, 4 × `LDSM.16.MT88.4` per k-step.

### Performance (my measurements — fill in)

| Kernel | N=1024 ms | N=1024 GFLOPS | N=2048 GFLOPS | vs 2d_vec | % of cuBLAS FP32 | max_rel_err |
|---|---|---|---|---|---|---|
| 2d_vec | TBD | TBD | TBD | 1.00× | TBD | TBD |
| db_reg | TBD | TBD | TBD | TBD | TBD | TBD |
| db_cpasync | TBD | TBD | TBD | TBD | TBD | TBD |
| warptile | TBD | TBD | TBD | TBD | TBD | TBD |
| wmma_naive (FP16) | TBD | TBD | TBD | TBD | TBD | TBD |
| wmma_fp16 (FP16) | TBD | TBD | TBD | TBD | TBD | TBD |
| cuBLAS FP32 | TBD | TBD | TBD | TBD | 100% | TBD |
| cuBLAS TF32 | TBD | TBD | TBD | TBD | TBD | TBD |
| cuBLAS FP16 | TBD | TBD | TBD | TBD | TBD | TBD |

Theoretical peaks for this GPU (derived from the GA102 whitepaper per-SM ratios, 16 SMs × 1.245–1.477 GHz): FP32 FFMA 5.1–6.0 TFLOPS, TF32 Tensor 5.1–6.0, FP16 Tensor with FP32 accumulate 10.2–12.1, FP16 accumulate 20.4–24.2.

### Nsight Compute evidence (N=1024 — fill in)

| Metric | 2d_vec | db_reg | db_cpasync | warptile | wmma_naive | wmma_fp16 |
|---|---|---|---|---|---|---|
| shared ld instructions (predicted 2,097,152 for FP32 kernels) | TBD | TBD | TBD | TBD | TBD | TBD |
| shared ld wavefronts (predicted ≈5.24M → ≈2.10M with warp tiling) | TBD | TBD | TBD | TBD | TBD | TBD |
| LDGSTS instructions (predicted 327,680 / 65,536) | TBD | TBD | TBD | TBD | TBD | TBD |
| FFMA thread inst. (= N³) / HMMA warp inst. (= N³/2048 = 524,288) | TBD | TBD | TBD | TBD | TBD | TBD |
| stall long_scoreboard (per issue) | TBD | TBD | TBD | TBD | TBD | TBD |
| stall barrier (per issue) | TBD | TBD | TBD | TBD | TBD | TBD |
| FMA / tensor pipe utilization % | TBD | TBD | TBD | TBD | TBD | TBD |
| achieved occupancy % | TBD | TBD | TBD | TBD | TBD | TBD |

### Correctness tooling

- `make racecheck`: normal build clean (TBD); `-DSKIP_DB_SYNC` (db_reg without its only barrier) → hazards (TBD).
- `make cpwait-bug`: `-DSKIP_CP_WAIT` (no `cp.async.wait_group`) → result TBD; racecheck / synccheck TBD.
- `make emu`: CPU-thread emulation of all six kernels (no GPU needed) — all OK at N=128/256; the no-wait and no-sync builds produce WRONG results.

**Finding (fill in after measuring):** Lesson 5's 2d_vec serialized load → wait → sync → compute. Double buffering cut long-scoreboard stalls from [X] to [Y] and barriers per k-step from 2 to 1, giving [Z]%. cp.async removed the staging registers (129 → 117) and STS instructions [...]. Warp tiling kept instruction counts identical but cut shared-memory wavefronts from [A] to [B]. Best FP32 kernel: [C] GFLOPS = [D]% of cuBLAS. Tensor Cores (FP16 in, FP32 accumulate): wmma_naive [E] GFLOPS (8 FLOP/B, starved), wmma_fp16 [F] GFLOPS vs cuBLAS FP16 [G]. Next: lesson 7 — driver API, contexts, modules, and the kernel launch path.
