// bank.cu  —  共享内存的 bank 冲突
//
// 对应笔记：docs/gpu-execution-model.md §6
// 编译运行： nvcc -O3 bank.cu -o bank && ./bank
//
// 关键：下标写成 (t*stride + i) 而不是 (t*stride)，是为了不让编译器
//       把循环里那个 load 提到循环外（循环不变量），否则测不出差别。

#include <cstdio>
#include <cuda_runtime.h>

// 一个 warp 的 32 个线程，各自反复读 buf[t * stride]
// stride 决定了这 32 个地址落在哪些 bank 上
__global__ void bank_test(float* out, int stride, int iters)
{
    __shared__ float buf[1024];
    int t = threadIdx.x;
    for (int i = t; i < 1024; i += blockDim.x) buf[i] = (float)i;   // 填满
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iters; i++)
        sum += buf[(t * stride + i) & 1023];  // ← 唯一变量：stride（+i 是为了不让编译器把 load 提到循环外）
    out[t] = sum;
}

__global__ void burn(float* o){ float v=threadIdx.x; for(long i=0;i<3000000L;i++) v=v*0.999999f+1.0f; o[threadIdx.x]=v; }

int main()
{
    float* d; cudaMalloc(&d, 1024*sizeof(float));
    float* db; cudaMalloc(&db, 4096);
    for(int i=0;i<200;i++) burn<<<14,1024>>>(db);   // 把频率拉满
    cudaDeviceSynchronize();

    const int ITERS = 20000;
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("一个 warp（32 线程）反复读共享内存，只改 stride\n");
    printf("每轮 %d 次循环，共 32 个 warp 同时跑\n\n", ITERS);
    printf("%8s %12s %10s %14s %10s\n","stride","地址模式","冲突倍数","耗时(ms)","相对s=1");
    printf("---------------------------------------------------------------------------------\n");
    double base = 0;
    for (int s : {0, 1, 2, 4, 8, 16, 32}) {
        bank_test<<<1,1024>>>(d, s, 100);      // 热身
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        bank_test<<<1,1024>>>(d, s, ITERS);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms=0; cudaEventElapsedTime(&ms,e0,e1);
        if (s==1) base = ms;

        // 算这个 stride 下一圈 32 个线程会撞几次 bank
        int hits[32] = {0};
        for (int t = 0; t < 32; t++) {
            int addr = (t*s) & 1023;
            hits[(addr/1) % 32]++;              // 一个 float 就是一个字，地址即字索引
        }
        int worst = 0; for (int b = 0; b < 32; b++) if (hits[b] > worst) worst = hits[b];

        const char* pat = (s==0) ? "全部同一地址" : (s==1) ? "连续(0..31)" : "跨步";
        printf("%8d %12s %10d %14.3f %9.2fx\n",
               s, pat, (s==0?1:worst), ms, ms/base);
    }
    return 0;
}
