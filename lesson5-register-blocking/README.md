## Lesson 5 — SGEMM register blocking (thread coarsening)

RTX 2050 Laptop (GA107, sm_86, 16 SMs, 4 GB), WSL2 Ubuntu, CUDA 13.5, ncu 2026.4. FP32, row-major, N × N.
Timing: cudaEvent, 0.5 s GPU wake-up, 1 warm-up + 10 timed runs, average. Every kernel checked against CPU i-k-j (N ≤ 2048) or cuBLAS (N > 2048), max_rel_err < 1e-3.

| Kernel | Idea | Threads/block | C per thread | Regs/thread (`-Xptxas -v`) | Blocks/SM | Occupancy theor. / achieved |
|---|---|---|---|---|---|---|
| `sgemm_tile16` | shared-memory tiling (lesson 4 baseline) | 256 | 1 | 37, 0 spill | 6 | 100% / 99.03% |
| `sgemm_1d_tm8` | 1D blocktiling, 64×64 block tile, BK=8 | 512 | 8 | 48, 0 spill | 2 | 66.67% / 64.81% |
| `sgemm_2d_8x8` | 2D blocktiling, 128×128 block tile, outer product in registers | 256 | 64 | 128, 0 spill | 2 | 33.33% / 30.37% |
| `sgemm_2d_vec` | 2D + float4 global loads/stores + transposed A tile in smem | 256 | 64 | 117, 0 spill | 2 | 33.33% / 30.95% |

All register / smem / blocks-per-SM numbers agree across ptxas, `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and ncu. SASS instruction counts (`make sass`) match the reference exactly.

### Performance (my measurements, 2026-11-25)

`make run`, one run (speedups computed within the same run):

| Kernel | N=1024 ms | N=1024 GFLOPS | N=2048 GFLOPS | vs tile16 (1024) | % of cuBLAS (1024) | max_rel_err |
|---|---|---|---|---|---|---|
| tile16 | 4.546 | 472.4 | 445.1 | 1.00× | 11.2% | 3.54e-07 |
| 1d_tm8 | 1.564 | 1372.9 | 1304.3 | 2.91× | 32.5% | 3.54e-07 |
| 2d_8x8 | 0.871 | 2466.6 | 2516.6 | 5.22× | 58.3% | 3.54e-07 |
| 2d_vec | 0.689 | 3118.1 | 2984.9 | 6.60× | 73.7% | 3.54e-07 |
| cuBLAS | 0.508 | 4230.6 | 3757.7 | 8.95× | 100% | 2.27e-06 |

`make sweep` (GFLOPS):

| Kernel | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|
| tile16 | 509.6 | 464.7 | 446.3 | 429.8 |
| 1d_tm8 | 1411.8 | 1334.0 | 1329.3 | 1291.5 |
| 2d_8x8 | 1860.9 | 2426.4 | 2444.4 | 2422.7 |
| 2d_vec | 2361.5 | 3095.5 | 2895.8 | 2730.1 |
| cuBLAS | 2977.1 | 4235.1 | 3728.0 | 2545.2 |

- N=512: the 2D kernels drop because 128×128 tiles give only 16 blocks for 16 SMs (half a wave), as predicted.
- N=4096: cuBLAS drops again (lesson 4: 2187.8) while my kernels lose only 0–12%, so whole-GPU throttling is unlikely to be the only cause; cuBLAS avg 54.0 ms vs best 49.9 ms (high variance). Cause not verified. One run at one size — not a claim that 2d_vec beats cuBLAS.
- tile16 measured 472.4 here vs 527.6 in lesson 4 (−10.5%) while cuBLAS was unchanged; cause not investigated.

### Nsight Compute evidence (N=1024, clocks locked at 1.24 GHz)

| Metric | tile16 | 1d_tm8 | 2d_8x8 | 2d_vec |
|---|---|---|---|---|
| shared ld instructions | 41,943,040 | 12,582,912 | 2,097,152 | 2,097,152 |
| shared ld wavefronts | 50,356,561 | 20,977,376 | 10,485,767 | 10,485,984 |
| wavefronts per shared ld instruction | 1.20 | 1.67 | 5.00 | 5.00 |
| shared bank conflicts ld / st | 15,831 / 147,179 | 4,433 / 112,867 | 4,194,304 / 0 | 4,194,304 / 262,144 |
| FFMA thread instructions | 1,073,741,824 | 1,073,741,824 | 1,073,741,824 | 1,073,741,824 |
| FFMA share of all warp instructions | 24.0% | 58.4% | 86.4% | 90.1% |
| global ld requests / sectors/request | 4,194,304 / 3.98 | 1,048,576 / 4.00 | 524,288 / 3.90 | 131,072 / 15.62 |
| DRAM read | 274.62 MB | 67.31 MB | 18.39 MB | 17.71 MB |
| L1/TEX throughput % | 97.12 | 81.87 | 62.41 | 78.15 |
| Mem Pipes Busy % | 97.00 | 81.70 | 31.16 | 31.52 |
| SM throughput % | 97.00 | 81.70 | 47.57 | 57.18 |
| Warp cycles per issued instruction | 35.36 | 19.46 | 7.62 | 6.52 |
| No Eligible % | 66.40 | 60.06 | 52.24 | 42.47 |

Shared ld/st instruction counts, FFMA = N³, global requests, registers and occupancy all match the pre-run predictions exactly.

**Finding.** Register blocking cut shared-memory load instructions from 41.9M to 2.1M (20×) with FFMA fixed at N³; Mem Pipes Busy fell from 97% to 31% and tile16's MIO-throttle stall (16.2 of 35.4 cycles per instruction) is gone. 2d_vec reaches 3118.1 GFLOPS, 6.6× tile16 and 73.7% of cuBLAS, despite 33% occupancy.

**What the prediction got wrong.** I predicted the transposed A tile would remove a 2-way bank conflict, so 2d_vec would need fewer shared-load wavefronts than 2d_8x8. Measured: identical (5.00 per `LDS.128`). A model that fits all four kernels exactly (my inference, not documented by NVIDIA): `LDS.128` is processed per half-warp, at least one wavefront per half. The two A rows sit in different half-warps, so there was never a conflict; all conflicts come from B reads (16 threads reading 16 B each at a 32 B stride → 4 wavefronts per half-warp; 1,048,576 B-loads × 4 extra = 4,194,304 reported conflicts). The transpose's real gain is vectorized A loads plus float4 global loads (requests 524,288 → 131,072).

**Remaining gap / next.** 0.3125 shared wavefronts per FFMA is still above the 0.25 budget; ncu still reports "dominated by L1". Fix: have each half-warp read 256 contiguous bytes of B (split each thread's 8 columns into two 4-wide pieces) — which is the warp-tiling layout of lesson 6. Plus no double buffering with only ~3.7 warps per scheduler (No Eligible 42–52%).

Not yet run: compute-sanitizer racecheck/memcheck for this lesson.
