# GPU 执行模型参数测量
# 用法： make        编译全部
#        make run    全部跑一遍（大概 1 分钟）
#        make clean

CUDA ?= /usr/local/cuda
NVCC  = $(CUDA)/bin/nvcc
FLAGS = -O3

TARGETS = specs flops stride bank occupancy sched

all: $(TARGETS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

run: all
	@echo "########## 1. 设备参数 ##########"; ./specs
	@echo; echo "########## 2. FP32 峰值 -> ALU 数量 ##########"; ./flops
	@echo; echo "########## 3. 访存带宽 ##########"; ./stride
	@echo; echo "########## 4. bank 冲突 ##########"; ./bank
	@echo; echo "########## 5. 占用率由什么限制 ##########"; ./occupancy
	@echo; echo "########## 6. 调度开销 ##########"; ./sched

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
