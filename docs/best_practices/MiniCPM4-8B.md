# MiniCPM4/4.1-8B

## I. Environment Preparation

### 1.1 Hardware Requirements
The minimum number of GPUs required to deploy `MiniCPM4.1-8B` on the following hardware for each quantization is as follows:

| | BF16 | WINT8 | WINT4 | FP8 |
|-----|-----|-----|-----|-----|
|H800 80GB| 1 | 1 | 1 | 1 |
|A800 80GB| 1 | 1 | 1 | / |
|H20 96GB| 1 | 1 | 1 | 1 |
|L20 48GB| 1 | 1 | 1 | 1 |
|A30 40GB| / | 1 | 1 | / |
|A10 24GB| / | 1 | 1 | / |
|V100 32GB| / | 1 | 1 | / |

**Tips:**
1. MiniCPM4.1-8B is a dense 8B model — a single GPU is sufficient for inference at all supported quantization levels.
2. For hardware not listed in the table, you can estimate whether it can be deployed based on the GPU memory. BF16 requires ~16GB, WINT8 ~8GB, WINT4 ~4GB.

### 1.2 Install FastDeploy
- Installation: For detail, please refer to [FastDeploy Installation](../get_started/installation/README.md).
- Model Download: For detail, please refer to [Supported Models](../supported_models.md).

## II. How to Use

### 2.1 Basic: Launching the Service

**Example 1:** Deploying MiniCPM4.1-8B with WINT4 quantization

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --tensor-parallel-size 1 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 128
```

**Example 2:** Deploying MiniCPM4.1-8B with BF16 (full precision)

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --max-num-seqs 64
```

- `--quantization`: Quantization strategy. Options: `wint8` / `wint4` / `block_wise_fp8` (Hopper required). Omit for BF16.
- `--max-model-len`: Maximum number of tokens for the deployed service. MiniCPM4.1 supports up to 65,536 tokens with LongRoPE, but larger values increase GPU memory usage.

For more parameter meanings and default settings, see [FastDeploy Parameter Documentation](../parameters.md).

### 2.2 Sending Requests

After the service starts, send requests via the OpenAI-compatible API:

```bash
curl http://localhost:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM4.1-8B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 512
  }'
```

### 2.3 Advanced: How to Get Better Performance

#### 2.3.1 Correctly Set Parameters That Match the Application Scenario
Evaluate average input length, average output length, and maximum context length.
- Set `--max-model-len` according to the maximum context length. For example, if the average input length is 1000 and the output length is 4000, then it is recommended to set it to 8192.

#### 2.3.2 Prefix Caching
**Idea:** The core idea of Prefix Caching is to avoid repeated calculations by caching the intermediate calculation results of the input sequence (KV Cache), thereby speeding up the response speed of multiple requests with the same prefix. For details, refer to [prefix-cache](../features/prefix_caching.md).

**How to enable:**
Since version 2.2 (including the develop branch), Prefix Caching has been enabled by default.

#### 2.3.3 Chunked Prefill
**Idea:** This strategy splits the prefill stage request into small-scale sub-chunks, and executes them in batches mixed with the decode request. For details, please refer to [Chunked Prefill](../features/chunked_prefill.md).

**How to enable:**
Since version 2.2 (including the develop branch), Chunked Prefill has been enabled by default.

#### 2.3.4 CudaGraph
**Idea:** CUDAGraph encapsulates GPU computing and memory operations into a re-executable graph, reducing CPU-GPU communication overhead and improving computing performance.

**How to enable:**
CUDAGraph has been enabled by default since version 2.3.

## Model Architecture Notes

MiniCPM4.1-8B uses μP (Maximal Update Parametrization) for training stability:
- **Embedding scaling**: Output scaled by `scale_emb` (12×)
- **Residual scaling**: Connections scaled by `scale_depth / √num_hidden_layers`
- **LM head scaling**: Input scaled by `hidden_size / dim_model_base`

These scaling factors are automatically read from the model's `config.json` and require no user configuration.

## FAQ
If you encounter any problems during use, please refer to [FAQ](./FAQ.md).
