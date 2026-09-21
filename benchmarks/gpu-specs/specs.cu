// specs.cu  —  查这张卡的执行模型参数（能直接查到的部分）
//
// 对应笔记：docs/gpu-execution-model.md §5.1
// 编译运行： nvcc -O3 specs.cu -o specs && ./specs
//
// 注意：这里查到的时钟是"基础频率"，不是加速频率。
// 要知道满载时真正跑多少，得一边跑 kernel 一边用 nvidia-smi 看。

#include <cstdio>
#include <cuda_runtime.h>

__global__ void topo()
{
    unsigned smid, warpid, laneid;
    asm("mov.u32 %0, %%smid;"   : "=r"(smid));
    asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    if (laneid == 0)
        printf("    block %u 的 warp %-2u -> SM %2u 上的第 %2u 个 warp 槽位\n",
               blockIdx.x, threadIdx.x/32, smid, warpid);
}

__global__ void empty_kernel() {}

int main()
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    auto A = [&](cudaDeviceAttr a) { int v = 0; cudaDeviceGetAttribute(&v, a, 0); return v; };

    printf("=== 1. 身份 ===\n");
    printf("设备名              %s\n", p.name);
    printf("计算能力            %d.%d   (warpSize = %d)\n", p.major, p.minor, p.warpSize);

    int sm  = A(cudaDevAttrMultiProcessorCount);
    int tpm = A(cudaDevAttrMaxThreadsPerMultiProcessor);
    printf("\n=== 2. 并行度（硬件天花板）===\n");
    printf("SM 数量             %d        <- 决定「最多点亮几个」\n", sm);
    printf("每 SM 最多线程      %d\n", tpm);
    printf("每 SM 最多 warp     %d        <- %d / warpSize\n", tpm / p.warpSize, tpm);
    printf("每 block 最多线程   %d\n", A(cudaDevAttrMaxThreadsPerBlock));
    printf("blockDim 上限       x=%d y=%d z=%d\n",
           A(cudaDevAttrMaxBlockDimX), A(cudaDevAttrMaxBlockDimY), A(cudaDevAttrMaxBlockDimZ));
    printf("gridDim  上限       x=%d y=%d z=%d\n",
           A(cudaDevAttrMaxGridDimX), A(cudaDevAttrMaxGridDimY), A(cudaDevAttrMaxGridDimZ));

    printf("\n=== 3. 片上存储 ===\n");
    printf("共享内存 / block    %d KB\n", A(cudaDevAttrMaxSharedMemoryPerBlock)/1024);
    printf("共享内存 / SM       %d KB\n", A(cudaDevAttrMaxSharedMemoryPerMultiprocessor)/1024);
    printf("寄存器   / SM       %d 个 32 位\n", A(cudaDevAttrMaxRegistersPerMultiprocessor));
    printf("L2 缓存             %d KB\n", A(cudaDevAttrMaxPersistingL2CacheSize) ? p.l2CacheSize/1024 : p.l2CacheSize/1024);
    printf("常量内存            %zu KB\n", p.totalConstMem/1024);

    printf("\n=== 4. 显存 ===\n");
    int clkMem = A(cudaDevAttrMemoryClockRate);            // kHz
    int bus    = A(cudaDevAttrGlobalMemoryBusWidth);       // bit
    double peak = 2.0 * clkMem * 1e3 * (bus/8) / 1e9;
    printf("容量                %.0f MB\n", p.totalGlobalMem/1048576.0);
    printf("位宽                %d bit        显存时钟 %d MHz\n", bus, clkMem/1000);
    printf("理论峰值带宽        %.1f GB/s     <- DDR，所以乘 2\n", peak);

    int clkSm = A(cudaDevAttrClockRate);
    printf("\n=== 5. 时钟 ===\n");
    printf("SM 基础频率         %d MHz        <- 不是满载频率！\n", clkSm/1000);
    printf("（满载频率要用 nvidia-smi 在跑 kernel 时看）\n");

    printf("\n=== 6. 每个 SM 能同时驻留几个 block ===\n");
    for (int b : {128, 256, 512, 1024}) {
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, empty_kernel, b, 0);
        printf("  blockDim=%-5d -> %d 个/SM，共 %4d 线程/SM（%d%% 占用）\n",
               b, nb, nb*b, nb*b*100/tpm);
    }

    printf("\n=== 7. 拓扑：block 怎么落到 SM 上（<<<4,128>>>）===\n");
    topo<<<4, 128>>>();
    cudaDeviceSynchronize();
    printf("  规律：一个 block 的所有 warp 必在同一个 SM；不同 block 分散到不同 SM。\n");

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("\nCUDA 错误：%s\n", cudaGetErrorString(e));
    return 0;
}
