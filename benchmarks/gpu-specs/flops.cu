// flops.cu  —  测 FP32 峰值 → 反推「这张卡有多少个 ALU」
//
// 对应笔记：docs/gpu-execution-model.md §5.2
// 编译运行： nvcc -O3 flops.cu -o flops && ./flops
//
// 原理：ALUS = 实测GFLOPS / (满载频率GHz x 2)
//       x2 是因为一次 FMA = 乘 + 加 = 2 个浮点运算
//
// 关键：CUDA 查到的时钟是基础频率，不是满载频率。
//       跑本程序时另开一个终端执行 nvidia-smi -q -d CLOCK 看真正跑多少。

#include <cstdio>
#include <cuda_runtime.h>

__global__ void empty_kernel() {}

// 纯计算，两条独立 FMA 链（一条链会被延迟卡住）
__global__ void fma_burn(float* out, int iters)
{
    float a = threadIdx.x, b = 1.000001f, c = 1e-7f;
    float d = threadIdx.x + 1, e = 1.000002f, f = 2e-7f;
    #pragma unroll 4
    for (int i = 0; i < iters; i++) {
        a = fmaf(a, b, c);  d = fmaf(d, e, f);
        a = fmaf(a, b, c);  d = fmaf(d, e, f);
    }
    if (a == -1.0f) out[0] = a + d;
}

static float g_ms;
static double dose(int grid, int block, int iters, float* d)
{
    fma_burn<<<grid, block>>>(d, 100);          // 热身
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    fma_burn<<<grid, block>>>(d, iters);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&g_ms, e0, e1);
    return (double)grid * block * iters * 4.0 * 2.0 / (g_ms * 1e-3) / 1e9;   // 每轮 4 次 FMA
}

int main()
{
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int sm = p.multiProcessorCount;
    float* d; cudaMalloc(&d, sizeof(float) * 4096);

    // 先把频率拉满：持续烧够 1.5 秒，否则 GPU 还在低频，测出来会偏低
    printf("预热中（持续烧 1.5 秒，把 GPU 频率拉满）...\n");
    {
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        float el = 0;
        while (el < 1500.0f) {
            fma_burn<<<sm * 4, 1024>>>(d, 20000);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&el, t0, t1);
        }
    }
    cudaDeviceSynchronize();

    const int ITERS = 100000;
    double peak = dose(sm * 4, 1024, ITERS, d);
    printf("=== 1. 峰值 FP32 吞吐 ===\n");
    printf("实测              %.1f GFLOPS\n", peak);

    int clkBase = 0;
    cudaDeviceGetAttribute(&clkBase, cudaDevAttrClockRate, 0);   // kHz，基础频率
    printf("\n=== 2. 反推 ALU 数量 ===\n");
    printf("反推公式          ALUS = GFLOPS / (满载频率GHz x 2)\n");
    printf("用基础频率 %.0f MHz 算  ->  %.0f 个（会偏大）\n",
           clkBase/1000.0, peak*1e9/(clkBase*1e3*2.0));
    printf("用满载频率 1740 MHz 算 ->  %.0f 个   <- 这个才对\n",
           peak*1e9/(1740e6*2.0));
    printf("除以 %d 个 SM          ->  每 SM 约 %.0f 个 FP32 单元\n",
           sm, peak*1e9/(1740e6*2.0)/sm);
    printf("\n满载频率请另开终端跑 nvidia-smi -q -d CLOCK 确认。\n");

    printf("\n=== 3. 每 SM 放几个 warp 才能喂满 ALU ===\n");
    printf("（这个形状能看出调度器/分区的数量级）\n\n");
    printf("%10s %10s %14s %10s\n", "blockDim", "warp/SM", "GFLOPS", "相对峰值");
    printf("--------------------------------------------------\n");
    for (int B : {32, 64, 128, 256, 512, 1024}) {
        double gf = dose(sm, B, ITERS, d);
        printf("%10d %10d %14.1f %9.1f%%\n", B, B/32, gf, gf/peak*100);
    }
    printf("\n读法：线性段说明 ALU 还没喂满；拐平后加 warp 没用。\n");

    cudaFree(d);
    return 0;
}
