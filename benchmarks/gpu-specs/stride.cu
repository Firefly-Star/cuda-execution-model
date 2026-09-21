// stride.cu  —  测「访问模式」对带宽的影响（显存侧）
//
// 对应笔记：docs/gpu-execution-model.md §5.3
// 编译运行： nvcc -O3 stride.cu -o stride && ./stride
//
// 两个实验：
//   A. 同一份数据反复读 —— 看数据能不能装进 L2 差多少
//   B. 同样的数据量、同样的线程数，只改相邻线程的地址差 —— 看合并访存

#include <cstdio>
#include <cuda_runtime.h>

__global__ void warmup(float* o){ float v=threadIdx.x; for(long i=0;i<3000000L;i++) v=v*0.999999f+1.0f; o[threadIdx.x]=v; }

// A. 反复扫读一个数组：小了命中 L2，大了只能去 DRAM
__global__ void readbench(const float* __restrict__ x, int m, int passes, float* out)
{
    long tid = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x*blockDim.x;
    float acc = 0.0f;
    for (int p = 0; p < passes; p++)
        for (long i = tid; i < m; i += stride) acc += x[i];
    if (acc == -1.0f) out[0] = acc;
}

// B. 相邻线程的地址差 = s（对 N 取模，保证元素和线程数都不变）
__global__ void copy_perm(const float* __restrict__ in, float* __restrict__ out, long N, int s)
{
    long t = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= N) return;
    long i = (t * (long)s) % N;
    out[i] = in[i];
}

static float *dx, *dy, *dw;

template <typename F>
static double timeit(F launch)
{
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); launch();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    return ms;
}

int main()
{
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);

    // 分配必须在预热之前！否则 warmup 会拿 NULL 指针去写，
    // 触发非法访存后整个 CUDA context 报废，后面所有调用都静默失败。
    cudaMalloc(&dx, 4<<20); cudaMalloc(&dy, 4<<20); cudaMalloc(&dw, 4096);
    cudaMemset(dx, 0, 4<<20);

    printf("预热中（持续烧 1.5 秒把频率拉满）...\n");
    {
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        float el = 0;
        while (el < 1500.0f) {
            warmup<<<p.multiProcessorCount*4, 1024>>>(dw);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&el, t0, t1);
        }
    }

    // ---------- A. L2 能装下多少，差多少 ----------
    printf("\n=== A. 反复读同一个数组：数据装得进 L2 吗 ===\n");
    printf("L2 = %d KB\n", p.l2CacheSize/1024);
    printf("总读取量固定 16 GB\n\n");
    printf("%12s %12s %14s %14s\n", "数组大小", "扫描轮数", "耗时(ms)", "读取GB/s");
    printf("--------------------------------------------------------\n");
    long TOTAL = 4000000000L;
    for (int m : {65536, 262144, 1048576}) {
        int passes = (int)(TOTAL/m);
        readbench<<<56,256>>>(dx, m, 10, dy); cudaDeviceSynchronize();
        double ms = timeit([&]{ readbench<<<56,256>>>(dx, m, passes, dy); });
        printf("%9d KB %12d %14.1f %14.1f\n",
               m*4/1024, passes, ms, TOTAL*4.0/(ms*1e-3)/1e9);
    }

    // ---------- B. 合并 vs 跨步 ----------
    printf("\n=== B. 元素数/线程数/总访存量固定，只改相邻线程的地址差 ===\n\n");
    long N = 1L<<22;                       // 4M 元素 = 16 MB，远超 L2
    cudaFree(dx); cudaFree(dy);
    cudaMalloc(&dx, 4*N); cudaMalloc(&dy, 4*N);
    cudaMemset(dx, 0, 4*N);
    int grid = (int)((N+255)/256);
    printf("%6s %16s %14s %14s %10s\n","步长s","warp覆盖跨度","耗时(ms)","有效带宽","相对s=1");
    printf("----------------------------------------------------------------------\n");
    double base = 0;
    for (int s : {1,2,4,8,16,32}) {
        copy_perm<<<grid,256>>>(dx,dy,N,s); cudaDeviceSynchronize();
        double ms = timeit([&]{ copy_perm<<<grid,256>>>(dx,dy,N,s); });
        double gb = 2.0*N*4/(ms*1e-3)/1e9;
        if (s==1) base = gb;
        printf("%6d %13d B %14.3f %11.1f GB/s %9.2fx\n",
               s, 31*s*4, ms, gb, base/gb);
    }
    printf("\n结论：s 越大越慢。因为一个 warp 的请求散进了越多的 32 字节扇区。\n");
    return 0;
}
