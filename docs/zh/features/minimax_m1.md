# MiniMax-M1 模型部署指南

## 模型简介

MiniMax-M1 是全球首个开源的大规模混合注意力推理模型（Hybrid-Attention Reasoning Model）。

### 核心特性

- **混合注意力架构**: Lightning Attention + Standard Softmax Attention
  - 70 层 Lightning Attention（线性注意力）
  - 10 层 Standard Attention（标准注意力）
  - 混合模式：每8层为1组（7层Lightning + 1层Standard）
- **MoE 架构**: 32个专家，每token激活2个专家
- **超长上下文**: 支持最高 100万 token 上下文
- **模型规模**: 总参数 456B，激活参数 45.9B

### 硬件需求

| 配置 | 要求 |
|------|------|
| 推荐配置 | 8x A100/H100 80GB (FP16) |
| 最低配置 | 4x A100 80GB + EP并行（需要量化） |

## 支持的功能

- [x] FP16 推理
- [x] 混合注意力机制
- [x] MoE 专家路由
- [x] 长上下文支持（最大 10M tokens）
- [ ] INT4/INT8 量化（开发中）
- [ ] Prefix Caching（开发中）

## 快速开始

### 安装依赖

```bash
pip install fastdeploy
```

### 模型加载

```python
import fastdeploy as fd

# 使用配置加载模型
option = fd.RuntimeOption()
option.use_paddle_inference()

# 配置模型路径
model_dir = "/path/to/MiniMax-M1-40k"
option.set_model_path(
    model_dir="",  # 模型目录
    model_file="", # 模型文件
    params_file="" # 参数文件
)

# 创建推理运行时
runtime = fd.Runtime(option)
```

### 配置参数

```json
{
  "model_type": "minimax_m1",
  "architectures": ["MiniMaxM1ForCausalLM"],
  "hidden_size": 6144,
  "intermediate_size": 9216,
  "num_hidden_layers": 80,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 200064,
  "num_local_experts": 32,
  "num_experts_per_tok": 2,
  "attn_type_list": [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
  "max_position_embeddings": 10240000
}
```

> 注意: `attn_type_list` 中 0 表示 Lightning Attention，1 表示 Standard Attention。每8层为一组循环。

### 推理示例

```python
# 构建输入
prompt = "请介绍一下MiniMax-M1模型的主要特点。"
inputs = fd.DataInput(texts=[prompt])

# 执行推理
results = runtime.predict(inputs)

# 获取结果
for result in results.text:
    print(result)
```

## 批量推理

```python
prompts = [
    "什么是Lightning Attention?",
    "MiniMax-M1支持多长的上下文?",
    "解释一下MoE架构"
]

inputs = fd.DataInput(texts=prompts)
results = runtime.predict(inputs)

for i, result in enumerate(results.text):
    print(f"Q{i+1}: {prompts[i]}")
    print(f"A{i+1}: {result}\n")
```

## 性能调优

### 1. 内存优化

对于显存受限的环境，可以使用以下策略：

```python
option = fd.RuntimeOption()
option.set_memory_optimize(level=1)  # 启用内存优化
```

### 2. 多卡并行

对于大模型推理，建议使用张量并行：

```python
option = fd.RuntimeOption()
option.set_gpu_tensor_parallel(
    tensor_parallel_size=4  # 使用4卡并行
)
```

### 3. Batch Size 调优

根据实际吞吐量需求调整 batch size：

```python
# 小batch低延迟
runtime = fd.Runtime(option, batch_size=1)

# 大batch高吞吐
runtime = fd.Runtime(option, batch_size=32)
```

## 注意事项

1. **显存要求**: FP16 推理至少需要 160GB GPU 显存
2. **长上下文**: 使用长上下文时注意显存消耗
3. **MoE 路由**: 专家路由在大 batch 下性能更优
4. **混合注意力**: Lightning Attention 层可显著降低长序列计算成本

## 常见问题

### Q: 模型加载失败怎么办？

A: 检查以下几点：
- 显存是否充足（推荐 8x A100 80GB）
- 模型文件路径是否正确
- CUDA 版本是否匹配（需要 CUDA 11.8+）

### Q: 推理速度慢怎么优化？

A: 尝试以下方法：
- 使用 Lightning Attention 层处理长序列
- 增大 batch size
- 启用 TensorRT 加速
- 使用多卡并行

### Q: 支持量化部署吗？

A: 量化功能正在开发中，预计后续版本支持 INT4/INT8 量化。

## 相关文档

- [FastDeploy 快速开始](https://github.com/PaddlePaddle/FastDeploy#quick-start)
- [模型列表](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/model_support_list.md)
- [量化推理](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/quantization.md)

## 更新日志

- **v1.0.0** (2026-03): 初始版本，支持 FP16 推理和混合注意力机制