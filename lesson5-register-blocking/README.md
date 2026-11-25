## Lesson 5 — SGEMM register blocking (thread coarsening)

RTX 2050 Laptop (GA107, sm_86, 16 SMs, 4 GB), WSL2 Ubuntu, CUDA 13.5, ncu 2026.4. FP32, row-major, N × N.
Timing: cudaEvent, 0.5 s GPU wake-up, 1 warm-up + 10 timed runs, average. Every kernel checked against CPU i-k-j (N ≤ 2048) or cuBLAS (N > 2048), max_rel_err < 1e-3.

| Kernel | Idea | Threads/block | C per thread | Regs/thread (`-Xptxas -v`) | Theor. occupancy |
|---|---|---|---|---|---|
| `sgemm_tile16` | shared-memory tiling (lesson 4 baseline) | 256 | 1 | 37, 0 spill | TBD |
| `sgemm_1d_tm8` | 1D blocktiling, 64×64 block tile, BK=8 | 512 | 8 | 48, 0 spill | TBD |
| `sgemm_2d_8x8` | 2D blocktiling, 128×128 block tile, outer product in registers | 256 | 64 | 128, 0 spill | TBD |
| `sgemm_2d_vec` | 2D + float4 global loads/stores + transposed A tile in smem | 256 | 64 | 117, 0 spill | TBD |

### Performance (my measurements — fill in)

| Kernel | N=1024 ms | N=1024 GFLOPS | N=2048 GFLOPS | vs tile16 | % of cuBLAS |
|---|---|---|---|---|---|
| tile16 | TBD | TBD | TBD | 1.00× | TBD |
| 1d_tm8 | TBD | TBD | TBD | TBD | TBD |
| 2d_8x8 | TBD | TBD | TBD | TBD | TBD |
| 2d_vec | TBD | TBD | TBD | TBD | TBD |
| cuBLAS | TBD | TBD | TBD | TBD | 100% |

### Nsight Compute evidence (N=1024 — fill in)

| Metric | tile16 | 1d_tm8 | 2d_8x8 | 2d_vec |
|---|---|---|---|---|
| shared ld instructions (`smsp__inst_executed_op_shared_ld.sum`) | TBD | TBD | TBD | TBD |
| shared ld wavefronts | TBD | TBD | TBD | TBD |
| FFMA thread instructions (should all be N³ = 1,073,741,824) | TBD | TBD | TBD | TBD |
| global ld sectors/request | TBD | TBD | TBD | TBD |
| L1/TEX throughput % | TBD | TBD | TBD | TBD |
| SM throughput % | TBD | TBD | TBD | TBD |
| achieved occupancy % | TBD | TBD | TBD | TBD |

**Finding (fill in after measuring):** Lesson 4 showed tile16 was bound by shared-memory/LSU instruction throughput (L1/TEX 97.1%, Mem Pipes Busy 96.99%, DRAM only 47.6%). Register blocking cut shared-memory load instructions from [X] to [Y] while FFMA count stayed at N³, moving the bottleneck to [...]. Remaining gap to cuBLAS: no double buffering, 33% occupancy, [...]. Next: double buffering / warp tiling.
