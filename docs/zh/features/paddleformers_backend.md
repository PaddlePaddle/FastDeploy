# PaddleFormers 后端

PaddleFormers 后端是 FastDeploy 的 fallback 机制，允许快速部署 PaddleFormers 兼容的模型，无需等待 FastDeploy 原生实现。

## 环境安装

源码安装 PaddleFormers（通过 paddlefleet 自动安装 PaddlePaddle）：

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers

# 根据 CUDA 版本选择
# CUDA 12.6
pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
# CUDA 12.9
pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
# CUDA 13.0
pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

> **注意**：如果出现包不存在的情况，可以通过 `--index-url https://mirrors.ustc.edu.cn/pypi/simple` 解决

更多选项参考 [PaddleFormers 官方安装文档](https://github.com/PaddlePaddle/PaddleFormers?tab=readme-ov-file#installation)。

## 快速开始

### 在线服务模式

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8582 \
    --engine-worker-queue-port 18582 \
    --metrics-port 28582 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --kv-cache-ratio 0.75 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --model-impl paddleformers
```

测试：

```bash
curl -X POST "http://0.0.0.0:8582/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"messages": [{"role": "user", "content": "把李白的静夜思改写为现代诗"}]}'
```

### 离线推理模式

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    model_impl="paddleformers",
    max_model_len=32768,
)

sampling_params = SamplingParams(max_tokens=4096, temperature=0.7)
messages = [[{"role": "user", "content": "将李白的静夜思改为现代诗歌"}]]
outputs = llm.chat(messages, sampling_params)

for output in outputs:
    print(output.outputs.text)
```

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model-impl` | `auto` | **默认值**。优先原生实现，无则回退 PaddleFormers |
| `--model-impl` | `fastdeploy` | 仅用原生实现，无则报错 |
| `--model-impl` | `paddleformers` | 强制使用 PaddleFormers 后端 |

## 支持情况

| 模型类型 | 状态 |
|----------|------|
| Dense 文本生成 (Qwen3/Llama3/ERNIE) | ✅ 支持 |
| VLM 视觉语言模型 | 开发中 |
| MOE 混合专家模型 | 开发中 |

| 加速策略 | 状态 |
|----------|------|
| Tensor Parallel (TP) | ✅ |
| CUDA Graph | ✅ |
| Prefix Caching | ✅ |
| Chunked Prefill | ✅ |
| QKV/Gate+Up 融合 | ✅ |

## 注意事项

**暂不支持**：Expert Parallel (EP)、量化推理 (INT4/INT8)、Speculative Decoding

**性能说明**：PaddleFormers 后端性能优于原生 PaddleFormers，略低于 FastDeploy 原生实现。
