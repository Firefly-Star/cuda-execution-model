# gpu-specs

测这张卡（GTX 1650 / TU117 / CC 7.5）的执行模型参数。

配套笔记：[`docs/gpu-execution-model.md`](../../docs/gpu-execution-model.md)

## 跑法

```bash
make run
```

## 四个程序

| 文件 | 测什么 | 对应笔记 |
| --- | --- | --- |
| `specs.cu` | 能直接查到的参数：SM 数、每 SM 线程/warp 上限、共享内存、L2、显存位宽、block 到 SM 的落位规律 | §6.1 |
| `flops.cu` | 查不到的参数：**实测 FP32 峰值反推 ALU 数量**；外加「几个 warp 才能喂满 ALU」的扫描 | §6.2 |
| `stride.cu` | 访存带宽：① 数据装不装得进 L2 差多少 ② 合并 vs 跨步差多少 | §6.3 |
| `bank.cu` | 共享内存的 bank 冲突：一个 warp 的 32 个线程落在哪些 bank 上 | §7 |
| `occupancy.cu` | 占用率被什么限制：block 上限、寄存器压力、共享内存压力，以及四条约束的 min 公式 | §6.5 |
| `sched.cu` | 调度开销：block 调度的边际成本（能测）+ warp 切换的零开销（反证） | 应用节 |

## 测量前提（很重要）

**必须先插电。** 判据：

```bash
nvidia-smi --query-gpu=clocks.mem,clocks.sm --format=csv
```

- 空闲时 `clocks.mem = 405 MHz` 是正常的省电降频
- **满载时必须是 5000 MHz**，否则就是在性能墙后面测，数字全部偏低

另外 `flops.cu` 内部会先持续烧 1.5 秒把频率拉满再开始测——不预热的话 GPU 还在低频，实测会差将近一倍（1773 vs 3112 GFLOPS）。

## 本机实测结果

见笔记末尾的汇总表。
