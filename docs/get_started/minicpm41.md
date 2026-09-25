[简体中文](../zh/get_started/minicpm41.md)

# Deploy the MiniCPM4.1-8B Model

This document explains how to deploy MiniCPM4.1-8B in BF16 or with online WINT4/WINT8 quantization. Before starting the deployment, ensure that your hardware environment meets the following requirements:

- GPU Driver >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 1 x 48 GB NVIDIA GPU

For FastDeploy installation instructions, refer to the [Installation Guide](./installation/README.md).

## Minimal Validation Setup

Enter the FastDeploy repository and specify only the local model directory and the GPU used for testing. The scripts automatically use the project `.venv` and configure its CUDA runtime libraries:

```shell
cd /path/to/FastDeploy
export MODEL_PATH=/path/to/MiniCPM4.1-8B
export CUDA_VISIBLE_DEVICES=0
```

## Prepare the Model

### 1. Manual Download (Optional)

MiniCPM4.1-8B uses the Hugging Face Torch safetensors format. Run the following commands to download the model:

```shell
export MODEL_PATH=/path/to/MiniCPM4.1-8B
hf download openbmb/MiniCPM4.1-8B --local-dir "${MODEL_PATH}"
```

The `pad_token_id` in the model configuration is 2. Before deployment, ensure that `pad_token` in `tokenizer_config.json` is set to `</s>`, which corresponds to token ID 2. Run the following command to complete the configuration:

```shell
MODEL_PATH="${MODEL_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["MODEL_PATH"]) / "tokenizer_config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["pad_token"] = "</s>"
config_path.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

### 2. Automatic Download

Set the model name to `openbmb/MiniCPM4.1-8B`:

```shell
export MODEL_PATH=openbmb/MiniCPM4.1-8B
```

>💡 **Note**: If the path specified by `--model` does not exist in the current directory, FastDeploy queries AIStudio for a preset model with that name. When found, the model is downloaded automatically with resumable transfers. AIStudio is the default download source; use `FD_MODEL_SOURCE` and `FD_MODEL_CACHE` to configure the source and cache directory. For details, see [Model Download](../supported_models.md). If the model has already been downloaded, you can set `MODEL_PATH` to its local directory instead.

## Build and Minimal Regression

### Build the CUDA Operators

Build the CUDA operators and verify the InfLLM-V2 symbols:

```shell
bash tests/benchmarks/test_minicpm41.sh build
```

The script invokes `build.sh`. Set `FD_BUILDING_ARCS` when targeting a different GPU architecture. A successful run prints `PASS build` and verifies that `fastdeploy/model_executor/ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so` contains these symbols:

- `infllmv2_update_compressed_k`
- `infllmv2_select_blocks`
- `infllmv2_attention_forward`

### MiniCPM4.1, Thinking, and Quantization Tests

Test model registration, weight mapping, multi-token thinking, mixed thinking modes, WINT4/WINT8 online quantization, and the InfLLM-V2 backend:

```shell
bash tests/benchmarks/test_minicpm41.sh unit
```

Run correctness tests for the compiled CUDA operators:

```shell
bash tests/benchmarks/test_minicpm41.sh operators
```

A zero exit status indicates success.

### End-to-End Serving Tests

Run BF16, WINT4, WINT8, and InfLLM-V2 validation sequentially with one command:

```shell
bash tests/benchmarks/test_minicpm41.sh e2e
```

The E2E runner automatically selects temporary ports, starts and cleans up each server, and runs five cases from `test_minicpm41_serving.py` in every mode. Each mode passes when it reports `5 passed`.

To run everything from compilation through E2E validation:

```shell
bash tests/benchmarks/test_minicpm41.sh
```

## Start the Service

>💡 **Note**: The following command serves the BF16 model on a single GPU with prefix caching disabled.

Run the following command to start the service. For details about the startup options, refer to the [Parameter Guide](../parameters.md).

```shell
export CUDA_VISIBLE_DEVICES=0
export FD_ATTENTION_BACKEND=FLASH_ATTN

python -m fastdeploy.entrypoints.openai.api_server \
       --model "${MODEL_PATH}" \
       --served-model-name MiniCPM4.1-8B \
       --port 8180 --engine-worker-queue-port 8182 \
       --cache-queue-port 8183 --metrics-port 8181 \
       --tensor-parallel-size 1 \
       --max-model-len 8192 \
       --max-num-seqs 1 \
       --max-num-batched-tokens 128 \
       --no-enable-prefix-caching
```

### Online WINT4/WINT8

Use the same original BF16 checkpoint and add `--quantization wint4` or `--quantization wint8`. FastDeploy quantizes each Linear weight in memory after loading it; no converted checkpoint is needed or supported by this MiniCPM4.1 path. The other service options are identical to the BF16 command above.

After completing the minimal validation setup, start WINT4:

```shell
bash scripts/run_minicpm41_wint_server.sh wint4 "${MODEL_PATH}"
```

Replace `wint4` with `wint8` for online INT8 weight quantization. The script requires `MODEL_PATH` to point to a local model directory and automatically uses the project `.venv`, its CUDA runtime libraries, and safe single-GPU defaults. Additional server options can be appended to the command. If the ports are already in use, assign an unused set of API, metrics, engine queue, and cache queue ports. See [Online Quantization](../quantization/online_quantization.md) for the shared WINT behavior.

## End-to-End Performance and Compression Results

The following results were measured with the original MiniCPM4.1-8B BF16 checkpoint and one NVIDIA RTX A6000 (SM86, TP=1). All modes used `max_model_len=1024`, `max_num_seqs=1`, `max_num_batched_tokens=256`, prefix caching disabled, and 32 fixed GPU KV-cache blocks. The serving benchmark sent 16 random requests at concurrency 1 with nominal 128-token inputs and 64-token outputs; the identical completed workload in every mode contained 2,895 input and 1,010 output tokens.

| Mode | Output tok/s | E2E speedup | Mean TTFT | Mean TPOT | TPOT speedup | Resident GPU memory | Memory compression | Memory reduction | Estimated parameter storage | Parameter compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 37.35 | 1.00x | 136.98 ms | 24.45 ms | 1.00x | 16,858 MiB (16.46 GiB) | 1.00x | 0.00% | 15.25 GiB | 1.00x |
| Online WINT4 (INT4 weight) | 96.72 | **2.59x** | 146.50 ms | 7.86 ms | 3.11x | 6,514 MiB (6.36 GiB) | **2.59x** | 61.36% | 4.66 GiB | 3.28x |
| Online WINT8 (INT8 weight) | 64.48 | **1.73x** | 140.23 ms | 13.24 ms | 1.85x | 10,416 MiB (10.17 GiB) | **1.62x** | 38.21% | 8.19 GiB | 1.86x |
| W4AFP8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

The end-to-end speedup is `quantized output throughput / BF16 output throughput`. Because all three runs produced the same 1,010 output tokens, it is also the inverse benchmark-duration ratio. Memory compression is `BF16 resident GPU memory / quantized resident GPU memory`; the 15 MiB idle GPU value was removed, and all modes reserved the same 32 KV-cache blocks. Estimated parameter storage is calculated from safetensors metadata and includes BF16 weights that are not quantized plus persistent BF16 scales. It is therefore different from resident GPU memory, which also includes runtime allocations and the KV cache. Online quantization slightly increased worker startup time from 25.41 seconds for BF16 to 27.41 seconds for WINT4 and 28.09 seconds for WINT8.

W4AFP8 does not have a valid result on this host. For dense Linear layers, the `w4afp8` CLI selection currently resolves to block-wise FP8, while the actual W4AFP8 implementation applies to `FusedMoE`; MiniCPM4.1-8B is dense. An attempted launch on SM86 failed during CUDA graph capture when DeepGEMM tried to build an SM90a FP8 kernel. Consequently, neither an end-to-end speedup nor an actual memory-compression ratio is reported. A nominal 4-bit weight payload may be described as roughly 4x smaller than BF16, but that theoretical payload ratio is not a runnable MiniCPM4.1 W4AFP8 result.

Reproduce the serving workload after starting each server with the settings above:

```shell
python benchmarks/benchmark_serving.py \
       --backend openai-chat \
       --base-url http://127.0.0.1:8180 \
       --endpoint /v1/chat/completions \
       --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}" \
       --dataset-name random --num-prompts 16 \
       --random-input-len 128 --random-output-len 64 \
       --max-concurrency 1 --seed 0 --save-result
```

## Send Requests to the Service

After you run the service startup command, the following terminal output indicates that the service has started successfully.

```shell
INFO api_server.py[line:1030] Launching metrics service at http://0.0.0.0:8181/metrics
INFO api_server.py[line:1033] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
INFO api_server.py[line:1034] Launching completion service at http://0.0.0.0:8180/v1/completions
[INFO] Starting gunicorn 26.0.0
[INFO] Listening at: http://0.0.0.0:8180
[INFO] Application startup complete.
```

FastDeploy provides a health check endpoint for checking the service status. If the following command returns `HTTP/1.1 200 OK`, the service has started successfully.

```shell
curl -i http://0.0.0.0:8180/health
```

Send a request using the following command. `enable_thinking=false` disables thinking mode.

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "MiniCPM4.1-8B",
  "messages": [
    {"role": "user", "content": "Rewrite the poem Quiet Night Thought by Li Bai as a modern poem."}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 64,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}' |  jq --indent 4 .
```

The response is as follows:

```json

{
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Certainly. Here is a modern-poetry adaptation of Li Bai's \"Quiet Night Thought\":\n\n**Night Thoughts**\n\nOutside the window, moonlight\npours down like flowing water\n\nA vast expanse of white\ncovers the cold window lattice\n\nA distant home\nflickers deep within memory\n\nA little bed\n"
            },
            "logprobs": null,
            "draft_logprobs": null,
            "prompt_logprobs": null,
            "finish_reason": "length",
            "speculate_metrics": null
        }
    ],
    "usage": {
        "prompt_tokens": 29,
        "total_tokens": 93,
        "completion_tokens": 64,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "image_tokens": 0,
            "video_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "image_tokens": 0
        }
    }
}
```

FastDeploy's service API is compatible with the OpenAI protocol. You can send a request using the following Python code:

```python
import openai

host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="MiniCPM4.1-8B",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Rewrite the poem Quiet Night Thought by Li Bai as a modern poem."},
    ],
    temperature=0,
    top_p=1,
    max_tokens=64,
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\n")
```

MiniCPM4.1 supports both thinking and non-thinking modes. Set `chat_template_kwargs.enable_thinking` to `true` in the request to enable thinking mode. You can also use `reasoning_max_tokens` to limit the number of thinking tokens used for the request.
