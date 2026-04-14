[简体中文](../zh/best_practices/MiniMax-M1.md)

# MiniMax-M1 Model

## I. Environment Preparation

### 1.1 Support Requirements

MiniMax-M1 support in FastDeploy uses a hybrid decoder stack:

- Standard full-attention layers run through the existing FastDeploy attention backend.
- Linear-attention layers use the Lightning Attention Triton kernels in `fastdeploy/model_executor/ops/triton_ops/lightning_attn.py`.
- Current first-pass support targets BF16 inference.

### 1.2 Installing FastDeploy

Installation process reference document [FastDeploy GPU Installation](../get_started/installation/nvidia_gpu.md)

## II. How to Use

### 2.1 Basics: Starting the Service

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

### 2.2 Quantized Deployment

MiniMax-M1 (456B params) requires quantization for practical deployment. Approximate GPU requirements:

| Mode | GPU Memory | Example Config |
|------|-----------|----------------|
| BF16 | ~912 GB | 12× A800-80GB, `--tensor-parallel-size 12` |
| FP8 | ~456 GB | 6× A800-80GB, `--tensor-parallel-size 6` |
| WINT4 | ~228 GB | 3× A800-80GB, `--tensor-parallel-size 4` |

```shell
# WINT4 quantization (recommended minimum)
python -m fastdeploy.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization wint4 \
    --tensor-parallel-size 4 \
    --port 8180 \
    --max-model-len 4096 \
    --max-num-seqs 4
```

### 2.3 Model Notes

- HuggingFace architecture: `MiniMaxText01ForCausalLM`
- Hybrid layer layout: 70 linear-attention layers and 10 full-attention layers
- MoE routing: 32 experts, top-2 experts per token

## III. Known Limitations

- This initial integration is focused on model structure and backend wiring.
- Low-bit quantization support still requires follow-up validation against MiniMax-M1 weights.
- Production validation should include GPU runtime checks for Lightning Attention decode/prefill paths.
