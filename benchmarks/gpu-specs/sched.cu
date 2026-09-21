// sched.cu  —  block 调度要花多少？warp 切换要花多少？
//
// 对应笔记：docs/blog/gpu-execution-model.md（调度开销那一节）
// 编译运行： nvcc -O3 sched.cu -o sched && ./sched
//
// block 调度：用空 kernel 扫 grid 大小，看边际成本。
// warp 切换：没法直接测——它不是一个"动作"，而是调度器每周期的选择本身。
//            它的"零开销"是靠峰值达到理论值反证出来的，见输出最后一段。

#include <cstdio>
#include <cuda_runtime.h>

__global__ void empty() {}                       // 什么都不干，只留调度成本

__global__ void burn(float* o)                    // 用来把频率拉满
{
    float v = threadIdx.x;
    for (long i = 0; i < 3000000L; i++) v = v * 0.999999f + 1.0f;
    o[threadIdx.x] = v;
}

__global__ void fma_burn(float* out, int iters)  // 测峰值用
{
    float a = threadIdx.x, b = 1.000001f, c = 1e-7f;
    float d = threadIdx.x + 1, e = 1.000002f, f = 2e-7f;
    for (int i = 0; i < iters; i++) {
        a = fmaf(a, b, c);  d = fmaf(d, e, f);
        a = fmaf(a, b, c);  d = fmaf(d, e, f);
    }
    if (a == -1.0f) out[0] = a + d;
}

static float* d;

// 空 kernel 扫一遍 grid 大小，返回每次发射的平均耗时（微秒）
static double bench_empty(int grid, int REP)
{
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 200; i++) empty<<<grid, 256>>>();   // 每档都充分热身
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < REP; i++) empty<<<grid, 256>>>();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    return ms * 1000.0 / REP;
}

int main()
{
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    cudaMalloc(&d, 4096);

    printf("预热：持续烧 1.5 秒把频率拉满...\n");
    {
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0); float el = 0;
        while (el < 1500.0f) {
            burn<<<sm * 4, 1024>>>(d);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&el, t0, t1);
        }
    }

    printf("\n=== 1. block 调度：空 kernel 只改 block 数 ===\n");
    printf("SM 数 = %d，每个 block 256 线程\n\n", sm);
    printf("%12s %14s %18s\n", "block 数", "耗时(us)", "每 block 边际(ns)");
    printf("--------------------------------------------------\n");
    double first = 0;
    for (int g : {1, 14, 56, 256, 1024, 4096, 16384, 65536, 262144, 1048576}) {
        int rep = g > 200000 ? 300 : (g > 16384 ? 800 : 3000);
        double t = bench_empty(g, rep);
        if (g == 1) first = t;
        printf("%12d %14.2f %18.2f\n", g, t, g > 1 ? (t - first) * 1000.0 / (g - 1) : 0.0);
    }
    printf("\n读法：\n");
    printf("  1 个 block 那几微秒是主机侧 launch 成本，跟 block 数无关。\n");
    printf("  大 grid 区的边际成本约 2 ns/block，要到 6.5 万块以上才看得见。\n");
    printf("  而且这是空 kernel——真实 kernel 里调度和计算重叠，更收不到这笔钱。\n");

    printf("\n=== 2. warp 切换：反证 ===\n");
    const int IT = 100000;
    fma_burn<<<sm * 4, 1024>>>(d, 1000); cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    fma_burn<<<sm * 4, 1024>>>(d, IT);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    double gf = (double)sm * 4 * 1024 * IT * 4.0 * 2.0 / (ms * 1e-3) / 1e9;
    printf("  实测 FP32 峰值       %.1f GFLOPS\n", gf);
    printf("  换算成每 SM 每周期    %.2f 条 warp-FMA\n", gf * 1e9 / 2.0 / (1.74e9) / sm / 32.0);
    printf("  理论值               2.00 条（= 64 个 FP32 单元 / 32 lane）\n");
    printf("\n  跑满了 = 8 个 warp 交替发射没有损耗。\n");
    printf("  如果每次切换要花一个周期，这里就不可能到 2.00。\n");

    cudaFree(d);
    return 0;
}
