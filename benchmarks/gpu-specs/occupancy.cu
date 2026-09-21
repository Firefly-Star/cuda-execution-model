// occupancy.cu  —  占用率到底被什么限制住了
//
// 对应笔记：docs/gpu-execution-model.md §6.5
// 编译运行： nvcc -O3 occupancy.cu -o occupancy && ./occupancy
//
// 三条独立的约束，任何一条都能成为瓶颈：
//   a) 每 SM 的线程数上限
//   b) 寄存器文件（65536 个）
//   c) 共享内存（64 KB / SM）

#include <cstdio>
#include <cuda_runtime.h>

// ---- 用于测寄存器压力：96 个元素的数组全展开在寄存器里，编译器压不掉 ----
#define BODY \
    float v[96]; \
    _Pragma("unroll") for (int i=0;i<96;i++) v[i] = threadIdx.x*0.001f + (float)i; \
    _Pragma("unroll") for (int k=0;k<20;k++) \
        _Pragma("unroll") for (int i=0;i<96;i++) v[i] = fmaf(v[i], v[(i+1)%96], 0.5f); \
    float s=0; \
    _Pragma("unroll") for (int i=0;i<96;i++) s+=v[i]; \
    if (s==-1.0f) out[0]=s;

// 只改「我打算用多少线程启动」，编译器据此分配寄存器预算
__global__ void __launch_bounds__(1024, 1) k_reg_1024(float* out) { BODY }
__global__ void __launch_bounds__( 256, 1) k_reg_256 (float* out) { BODY }
__global__ void __launch_bounds__( 128, 1) k_reg_128 (float* out) { BODY }

// ---- 用于测共享内存压力 ----
__global__ void k_smem(float* out)
{
    extern __shared__ float s[];
    int t = threadIdx.x;
    s[t] = t * 0.5f;
    __syncthreads();
    float v = 0;
    for (int i = 0; i < 32; i++) v += s[(t + i*7) % blockDim.x];
    out[t] = v;
}

__global__ void k_simple(float* out) { out[threadIdx.x] = threadIdx.x; }

int main()
{
    float* d; cudaMalloc(&d, 4096);
    int tpm=0, mpb=0, smemSM=0, smemBlk=0, maxBlk=0;
    cudaDeviceGetAttribute(&tpm,     cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&mpb,     cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&smemSM,  cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&smemBlk, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&maxBlk,  cudaDevAttrMaxBlocksPerMultiprocessor, 0);

    printf("=== 本机的四条约束 ===\n");
    printf("  每 SM 线程上限      %d\n", tpm);
    printf("  每 block 线程上限   %d      <- 和上面是【两条独立约束】，不是推出来的\n", mpb);
    printf("  寄存器文件 / SM     %d 个 32 位\n", 65536);
    printf("  共享内存 / SM       %d KB\n", smemSM/1024);
    printf("  每 SM 最多 block    %d\n", maxBlk);

    printf("\n=== 1. 一个 block 超过 %d 个线程会怎样 ===\n", mpb);
    for (int b : {1024, 1025, 2048}) {
        k_simple<<<1, b>>>(d);
        cudaError_t e = cudaGetLastError();
        printf("  <<<1, %4d>>>  ->  %s\n", b, e==cudaSuccess ? "启动成功" : cudaGetErrorString(e));
        if (e != cudaSuccess) { cudaGetLastError(); cudaDeviceSynchronize(); }
    }
    printf("  硬报错，不会静默截断；而且是同步返回的（不用等 synchronize）。\n");

    printf("\n=== 2. 寄存器限制占用率（同一个循环体，只改寄存器预算）===\n");
    printf("  %-28s %12s %14s %12s\n", "kernel", "寄存器/线程", "每 SM block", "线程/SM");
    struct { const char* n; const void* f; int blk; } rk[] = {
        {"__launch_bounds__(1024, 1)", (const void*)k_reg_1024, 1024},
        {"__launch_bounds__(256, 1)",  (const void*)k_reg_256,   256},
        {"__launch_bounds__(128, 1)",  (const void*)k_reg_128,   128},
    };
    for (auto& k : rk) {
        cudaFuncAttributes a; cudaFuncGetAttributes(&a, k.f);
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k.f, k.blk, 0);
        printf("  %-28s %12d %14d %10d (%.0f%%)\n",
               k.n, a.numRegs, nb, nb*k.blk, 100.0*nb*k.blk/tpm);
    }
    printf("  公式：能驻留的 block 数 = 65536 / (寄存器数 x blockDim)\n");
    printf("  注意第一行：占用率 100%% 的代价是寄存器被压到 64，可能在偷偷 spill。\n");

    printf("\n=== 3. 共享内存限制占用率（blockDim 固定 256）===\n");
    cudaFuncSetAttribute(k_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBlk);
    printf("  %-16s %14s %12s %10s\n", "每 block 共享内存", "每 SM block", "线程/SM", "占用率");
    for (int kb : {0, 16, 24, 32, 48}) {
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)k_smem, 256, kb*1024);
        printf("  %13d KB %14d %12d %9.0f%%\n", kb, nb, nb*256, 100.0*nb*256/tpm);
    }
    printf("  公式：能驻留的 block 数 = 64KB / 每 block 共享内存\n");

    printf("\n=== 4. 完整的约束 ===\n");
    printf("  每 SM 能驻留的 block 数 = min(\n");
    printf("      %d / blockDim,                    每 SM 线程上限\n", tpm);
    printf("      65536 / (寄存器数 x blockDim),     寄存器文件\n");
    printf("      %dKB / 每 block 共享内存,           共享内存\n", smemSM/1024);
    printf("      %d                                 每 SM 最多 block\n", maxBlk);
    printf("  )\n");
    printf("  四条里任何一条都能成为瓶颈，光看 blockDim 是算不出来的。\n");

    cudaFree(d);
    return 0;
}
