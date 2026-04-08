[English](../../best_practices/MiniMax-M1.md)

# MiniMax-M1 模型

## 一、环境准备

### 1.1 支持说明

FastDeploy 中的 MiniMax-M1 采用混合解码器结构：

- 全注意力层复用 FastDeploy 现有 Attention 后端。
- 线性注意力层使用 `fastdeploy/model_executor/ops/triton_ops/lightning_attn.py` 中的 Lightning Attention Triton kernel。
- 当前首版支持以 BF16 推理为主。

### 1.2 安装 FastDeploy

安装流程可参考 [FastDeploy GPU 安装文档](../get_started/installation/nvidia_gpu.md)

## 二、使用方式

### 2.1 基础启动命令

```shell
MODEL_PATH=/models/MiniMax-Text-01

python -m fastdeploy.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --max-model-len 32768 \
    --max-num-seqs 32
```

### 2.2 模型特性

- HuggingFace 架构名：`MiniMaxText01ForCausalLM`
- 层类型分布：70 层线性注意力 + 10 层全注意力
- MoE 路由：32 个专家，每个 token 选择 top-2 专家

## 三、当前限制

- 当前版本优先完成模型组网与后端接线。
- 各类低比特量化推理能力还需要结合真实权重进一步验证。
- Lightning Attention 的 prefill/decode 路径仍需在 GPU 环境完成端到端验证。
- 线性注意力的 KV history 当前使用实例变量存储，多请求并发场景下需迁移至 slot-based cache（已有 TODO 标注）。
