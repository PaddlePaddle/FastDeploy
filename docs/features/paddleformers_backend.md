# PaddleFormers Backend

The PaddleFormers backend is FastDeploy's fallback mechanism, enabling rapid deployment of PaddleFormers-compatible models without waiting for native FastDeploy implementations.

## Installation

Install PaddleFormers from source with paddlefleet (which automatically installs PaddlePaddle):

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers

# Choose based on your CUDA version
# CUDA 12.6
pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
# CUDA 12.9
pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
# CUDA 13.0
pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

> **Note**: If packages are not found, try adding `--index-url https://mirrors.ustc.edu.cn/pypi/simple`

For more options, refer to the [PaddleFormers Official Installation Guide](https://github.com/PaddlePaddle/PaddleFormers?tab=readme-ov-file#installation).

## Quick Start

### Online Serving Mode

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

Test:

```bash
curl -X POST "http://0.0.0.0:8582/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"messages": [{"role": "user", "content": "Write a haiku about programming"}]}'
```

### Offline Inference Mode

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    model_impl="paddleformers",
    max_model_len=32768,
)

sampling_params = SamplingParams(max_tokens=4096, temperature=0.7)
messages = [[{"role": "user", "content": "Write a short poem about artificial intelligence"}]]
outputs = llm.chat(messages, sampling_params)

for output in outputs:
    print(output.outputs.text)
```

## Parameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model-impl` | `auto` | **Default**. Prefers native implementation, falls back to PaddleFormers if unavailable |
| `--model-impl` | `fastdeploy` | Uses native implementation only, raises error if unavailable |
| `--model-impl` | `paddleformers` | Forces use of PaddleFormers backend |

## Support Status

| Model Type | Status |
|------------|--------|
| Dense Text Generation (Qwen3/Llama3/ERNIE) | ✅ Supported |
| VLM (Vision-Language Models) | In Development |
| MOE (Mixture of Experts) | In Development |

| Optimization | Status |
|--------------|--------|
| Tensor Parallel (TP) | ✅ |
| CUDA Graph | ✅ |
| Prefix Caching | ✅ |
| Chunked Prefill | ✅ |
| QKV/Gate+Up Fusion | ✅ |

## Limitations

**Not Yet Supported**: Expert Parallel (EP), Quantized Inference (INT4/INT8), Speculative Decoding

**Performance Note**: The PaddleFormers backend outperforms native PaddleFormers but is slightly slower than native FastDeploy implementations.
