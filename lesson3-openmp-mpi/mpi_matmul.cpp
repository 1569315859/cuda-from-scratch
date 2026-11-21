// mpi_matmul.cpp —— 用 MPI 多进程做矩阵乘：按行分块，分别测「通信」和「计算」时间
// 编译：mpicxx -O2 -std=c++17 mpi_matmul.cpp -o mpi_matmul
// 运行：mpirun -np 4 ./mpi_matmul [N] [check]     N 默认 1024；check 写 0 跳过正确性检查
//       进程数超过 CPU 核数时要加 --oversubscribe
#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 我是几号进程（0 ~ size-1）
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // 一共几个进程

    const int n = (argc > 1) ? atoi(argv[1]) : 1024;
    const bool check = (argc > 2) ? atoi(argv[2]) != 0 : true;
    // 每个进程分到 rows 行；n 不能被 size 整除时，把总行数向上补齐到 rows*size，多出的行补 0
    const int rows = (n + size - 1) / size;
    const int padded = rows * size;

    std::vector<float> A, C;                         // 完整的 A、C 只在 0 号进程上有
    std::vector<float> B((size_t)n * n);             // 每个进程都要一份完整的 B
    std::vector<float> A_local((size_t)rows * n);    // 我负责的那几行 A
    std::vector<float> C_local((size_t)rows * n);    // 我算出的那几行 C
    if (rank == 0) {
        A.assign((size_t)padded * n, 0.0f);
        C.assign((size_t)padded * n, 0.0f);
        std::vector<float> tmp((size_t)n * n);
        fill_random(tmp, 1);                         // 和 omp_matmul 用同样的种子
        std::copy(tmp.begin(), tmp.end(), A.begin());// 前 n 行是真数据，后面是补的 0
        fill_random(B, 2);
    }

    const int reps = 3;
    double best_total = 1e30, best_dist = 0, best_comp_max = 0, best_comp_min = 0, best_gather = 0;
    for (int r = 0; r < reps; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);                 // 大家站齐再开始计时
        double t0 = MPI_Wtime();

        // ---- 通信 1：把 B 广播给所有人，把 A 按行切块分发 ----
        MPI_Bcast(B.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(rank == 0 ? A.data() : nullptr, rows * n, MPI_FLOAT,
                    A_local.data(), rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // ---- 计算：每个进程只算自己的 rows 行（i-k-j 顺序）----
        double tc = MPI_Wtime();
        for (int i = 0; i < rows; ++i) {
            float* c = &C_local[(size_t)i * n];
            for (int j = 0; j < n; ++j) c[j] = 0.0f;
            for (int k = 0; k < n; ++k) {
                const float a = A_local[(size_t)i * n + k];
                const float* b = &B[(size_t)k * n];
                for (int j = 0; j < n; ++j) c[j] += a * b[j];
            }
        }
        tc = MPI_Wtime() - tc;                       // 我自己的纯计算时间
        MPI_Barrier(MPI_COMM_WORLD);                 // 等最慢的进程算完
        double t2 = MPI_Wtime();

        // ---- 通信 2：把各自算的行收集回 0 号进程 ----
        MPI_Gather(C_local.data(), rows * n, MPI_FLOAT,
                   rank == 0 ? C.data() : nullptr, rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        double t3 = MPI_Wtime();

        // 用 MPI_Reduce 求所有进程计算时间的最大值和最小值，看负载是否均衡
        double tc_max = 0, tc_min = 0;
        MPI_Reduce(&tc, &tc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tc, &tc_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0 && (t3 - t0) < best_total) {
            best_total = t3 - t0; best_dist = t1 - t0; best_gather = t3 - t2;
            best_comp_max = tc_max; best_comp_min = tc_min;
        }
    }

    if (rank == 0) {
        printf("N = %d, processes = %d, rows per process = %d\n", n, size, rows);
        printf("distribute (Bcast+Scatter): %9.1f ms\n", best_dist * 1e3);
        printf("compute (slowest / fastest): %8.1f / %.1f ms\n", best_comp_max * 1e3, best_comp_min * 1e3);
        printf("gather:                     %9.1f ms\n", best_gather * 1e3);
        printf("total:                      %9.1f ms   (%.2f GFLOPS)\n", best_total * 1e3, gflops(n, best_total));
        printf("communication share:        %9.1f %%\n", 100.0 * (best_dist + best_gather) / best_total);
        if (check) {
            std::vector<float> Aref((size_t)n * n), Ref((size_t)n * n);
            std::copy(A.begin(), A.begin() + (size_t)n * n, Aref.begin());
            matmul_ikj_serial(Aref.data(), B.data(), Ref.data(), n);
            double err = max_rel_err(C.data(), Ref.data(), (size_t)n * n);
            printf("max_rel_err vs serial: %.2e  %s\n", err, err < 1e-3 ? "OK" : "WRONG");
        }
    }
    MPI_Finalize();
    return 0;
}
