# cuda-execution-model

在一台笔记本上实测 NVIDIA GPU 的执行模型：**thread / warp / warp 调度器 / ALU / block / SM 这七层到底谁管什么**。

- 配套文章：[docs/blog/gpu-execution-model.md](docs/blog/gpu-execution-model.md)（面向初学者）
- 完整笔记：[docs/gpu-execution-model.md](docs/gpu-execution-model.md)（含全部原始数据和推算过程）

## 这份代码回答什么问题

- 一个 warp 32 个线程，为什么每个子分区只有 16 个 ALU？
- `bank 冲突` 是 warp 级、block 级还是 SM 级？
- 一个 SM 到底能放多少 warp？「1024 线程」这个上限是谁定的？
- grid 开几千个 block 有调度开销吗？warp 切来切去有开销吗？
- 数据装不装得进 L2 差多少？跨步访问差多少？

每一个结论都对应下面六个程序里的一段测量。

## 跑起来

```bash
cd benchmarks/gpu-specs
make run          # 六个全跑，大约 1 分钟
```

或者单独编某一个：

```bash
nvcc -O3 specs.cu -o specs && ./specs
```

## 六个程序

| 文件 | 测什么 |
| --- | --- |
| `specs.cu` | 能直接查到的参数：SM 数、每 SM 线程/warp 上限、共享内存、L2、显存位宽；外加 block 落到哪个 SM 的规律 |
| `flops.cu` | 查不到的参数：**实测 FP32 峰值反推 ALU 数量**；外加「每 SM 放几个 warp 才能喂满 ALU」的曲线 |
| `stride.cu` | 访存带宽：数据装不装得进 L2 差多少、合并访问 vs 跨步访问差多少 |
| `bank.cu` | 共享内存的 bank 冲突：一个 warp 的 32 个线程落在哪些 bank 上 |
| `occupancy.cu` | 占用率被什么限制：block 上限、寄存器压力、共享内存压力，以及四条约束的取 min 公式 |
| `sched.cu` | 调度开销：block 调度的边际成本（能测）+ warp 切换的零开销（用峰值反证） |

## ⚠️ 测量前提

**必须先插电，并且等 GPU 升频。** 这两条不满足的话，所有数字都会偏低。

判据是跑负载时看显存时钟：

```bash
nvidia-smi --query-gpu=clocks.mem,clocks.sm --format=csv
```

```
空闲：405 MHz      <- 省电降频，正常
满载：5000 MHz     <- 插电后应该看到这个
```

**GPU 也不会一开跑就到最高频率。** 实测差别很大：

```
直接开测                1773 GFLOPS
先持续烧 1.5 秒再测     3112 GFLOPS
```

所以 `flops.cu`、`stride.cu`、`occupancy.cu`、`sched.cu` 内部都会先热身够时间才开始计时。这一步不能省。

## 测试硬件

| 项 | 值 |
| --- | --- |
| GPU | GeForce GTX 1650 4GB（TU117，计算能力 7.5，14 个 SM） |
| 平台 | WSL2 (Ubuntu 22.04) + CUDA 13.3 |
| SM 频率 | 基础 1155 MHz / 满载 1740 MHz |
| 显存 | 4096 MB，128 bit，5001 MHz → 理论 160 GB/s |

**换个架构数字会变。** 尤其是 A100 / H100 那一档，每 SM 的线程上限是 2048 而不是 1024。文中凡是引用官方规格而不是实测的地方都有注明。

## 已知的坑

- **CUDA 查到的时钟是基础频率，不是满载频率。** `cudaDevAttrClockRate` 返回 1155 MHz，满载是 1740 MHz。用它反推 ALU 数量会得到 1347 这种荒谬的值。
- **「每 SM 1024 线程」和「每 block 1024 线程」是两条独立约束**，只是在这张卡上恰好相等。（A100/H100 上每 SM 是 2048，每 block 仍是 1024。）
- **warp 调度器数量测不出来。** 从吞吐曲线只能推测是 4 个；`ncu` 的指标按子分区统计，本来可以直接读，但 WSL2 拿不到 GPU 性能计数器权限。
- **共享内存是片上 SRAM，不是显存。** 「拆成多拍」发生在 LSU 到共享内存单元这一段，全程都在片上。
