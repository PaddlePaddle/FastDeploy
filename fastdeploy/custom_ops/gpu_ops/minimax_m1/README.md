# MiniMax-M1 Custom GPU Operations

本目录包含 MiniMax-M1 模型的自定义 GPU 算子实现，基于 CUDA 加速。

## 功能特性

- **Lightning Attention Kernel**: 高效的线性注意力实现
- **Block-wise 计算**: 将序列分块处理，降低显存占用
- **Causal Masking**: 使用指数衰减实现高效的因果注意力
- **增量推理支持**: 支持 KV cache 的增量更新

## 构建方法

### 前置条件

- CUDA 11.8+
- CMake 3.18+
- GCC/Clang 编译器

### 编译步骤

```bash
# 进入算子目录
cd fastdeploy/custom_ops/gpu_ops/minimax_m1

# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH="80;86;89;90" \
         -DCMAKE_INSTALL_PREFIX=/path/to/fastdeploy

# 编译
make -j8

# 安装
make install
```

### 使用 Docker（推荐）

```bash
# 使用 FastDeploy CUDA 镜像
docker run --gpus all -it registry.baidubce.com/paddlepaddle/fastdeploy:dev-gpu-cuda12.2-trt8.6 bash

# 在容器内编译
cd /opt/fastdeploy
mkdir -p build && cd build
cmake .. && make -j8 && make install
```

## 使用方法

### Python API

```python
import fastdeploy as fd
from fastdeploy.custom_ops.gpu_ops import minimax_m1

# Lightning Attention 会在模型推理时自动调用
model = fd.LLMModel("MiniMax-M1-40k")
result = model.predict("Hello world")
```

### C++ API

```cpp
#include "fastdeploy/custom_ops/gpu_ops/minimax_m1/lightning_attention.h"

// 在模型代码中调用
fastdeploy::custom_ops::LaunchLightningAttention(
    q_data, k_data, v_data, output_data,
    causal_mask, scale,
    batch_size, num_heads, num_kv_heads,
    seq_len, head_dim,
    cuda_stream
);
```

## 性能优化

### 1. Block Size 调优

```cpp
// 根据你的 GPU 和序列长度调整 block size
#define LA_BLOCK_SIZE 64  // 默认值，可调整为 32/64/128
```

### 2. 显存优化

- 减少 block size 可以降低显存占用
- 使用 half precision (FP16) 可以显著降低显存

### 3. 多卡并行

对于长序列，可以结合张量并行使用：

```bash
# 使用 4 卡并行
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 文件结构

```
minimax_m1/
├── CMakeLists.txt          # 构建配置
├── README.md               # 本文件
├── lightning_attention.h   # CUDA kernel 头文件
├── lightning_attention.cpp # CUDA kernel 实现
└── __init__.py             # Python 绑定
```

## 已知限制

1. **硬件要求**: 需要 Ampere 架构或更新的 GPU (RTX 30xx / A100+)
2. **显存要求**: 长序列需要大量显存，建议 24GB+ GPU
3. **序列长度**: 当前最大支持 100K tokens

## 常见问题

### Q: 编译失败怎么办？

A: 检查以下几点：
- CUDA 版本是否 >= 11.8
- GPU 驱动是否支持对应的 CUDA 架构
- 是否有足够的磁盘空间（编译产物约 100MB）

### Q: 运行时找不到库？

A: 设置 LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/path/to/fastdeploy/custom_ops/gpu_ops/minimax_m1:$LD_LIBRARY_PATH
```

## 贡献指南

欢迎提交 PR 改进算子实现！请确保：

1. 通过所有单元测试
2. 性能不劣于现有实现
3. 遵守代码风格规范

## 更新日志

- **v1.0.0** (2026-03): 初始版本，支持 Lightning Attention 前向传播