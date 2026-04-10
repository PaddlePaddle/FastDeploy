[简体中文](../zh/usage/kunlunxin_xpu_deployment.md)

## Supported Models
Note: The following models and deployment commands are valid only for version 2.5.0.
<details>
<summary><b>ERNIE-4.5-300B-A47B (32K, WINT8, 8 XPUs)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4 # Check the RDMA network card names of the machine using `xpu-smi topo -m`
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --engine-worker-queue-port 8124 \
    --metrics-port 8125 \
    --cache-queue-port 55996 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --data-parallel-size 1 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-300B-A47B (32K, WINT4, 4 XPUs)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3"  # or "4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3"  # or "4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2  # Check the RDMA network card names of the machine using `xpu-smi topo -m`
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --engine-worker-queue-port 8124 \
    --metrics-port 8125 \
    --cache-queue-port 55996 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization wint4 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --data-parallel-size 1 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-300B-A47B (32K, WINT4, 8 XPUs)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.95
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4  # Check the RDMA network card names of the machine using `xpu-smi topo -m`
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1
python -m fastdeploy.entrypoints.openai.api_server \
    --model /PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --engine-worker-queue-port 8124 \
    --metrics-port 8125 \
    --cache-queue-port 55996 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization wint4 \
    --gpu-memory-utilization 0.95 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --data-parallel-size 1 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-300B-A47B (128K, WINT4, 8 XPUs)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4  # Check the RDMA network card names of the machine using `xpu-smi topo -m`
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1
python -m fastdeploy.entrypoints.openai.api_server \
    --model /PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8123 \
    --engine-worker-queue-port 8124 \
    --metrics-port 8125 \
    --cache-queue-port 55996 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization wint4 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --data-parallel-size 1 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-21B-A3B (32K, BF16, 1 XPU)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-21B-A3B (32K, WINT8, 1 XPU)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-21B-A3B (32K, WINT4, 1 XPU)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-21B-A3B (128K, BF16, 1 XPU)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-21B-A3B (128K, WINT8, 1 XPU)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-21B-A3B (128K, WINT4, 1 XPU)</b> </summary>

**Quick Launch:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**Optimal Performance:**
```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "'${mtp_model_path}'"}'
```
</details>

<details>
<summary><b>ERNIE-4.5-0.3B (32K, BF16, 1 XPU)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```
</details>

<details>
<summary><b>ERNIE-4.5-0.3B (32K, WINT8, 1 XPU)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```
</details>

<details>
<summary><b>ERNIE-4.5-0.3B (128K, BF16, 1 XPU)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```
</details>

<details>
<summary><b>ERNIE-4.5-0.3B (128K, WINT8, 1 XPU)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```
</details>

<details>
<summary><b>ERNIE-4.5-300B-A47B-W4A8C8-TP4 (32K, W4A8, 4 XPUs)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3"  # or "4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "W4A8" \
    --gpu-memory-utilization 0.9
```
</details>

<details>
<summary><b>ERNIE-4.5-VL-28B-A3B (32K, WINT8, 1 XPU)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --quantization "wint8" \
    --max-model-len 32768 \
    --max-num-seqs 10 \
    --enable-mm \
    --mm-processor-kwargs '{"video_max_frames": 30}' \
    --limit-mm-per-prompt '{"image": 10, "video": 3}' \
    --reasoning-parser ernie-45-vl
```
</details>

<details>
<summary><b>ERNIE-4.5-VL-424B-A47B (32K, WINT8, 8 XPUs)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 8 \
    --quantization "wint8" \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-mm \
    --mm-processor-kwargs '{"video_max_frames": 30}' \
    --limit-mm-per-prompt '{"image": 10, "video": 3}' \
    --reasoning-parser ernie-45-vl \
    --gpu-memory-utilization 0.7
```
</details>

<details>
<summary><b>PaddleOCR-VL-0.9B (32K, BF16, 1 XPU)</b> </summary>

```bash
export FD_ENABLE_MAX_PREFILL=1
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8188 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 256
```
</details>

<details>
<summary><b>ERNIE-4.5-VL-28B-A3B-Thinking (128K, WINT8, 1 XPU)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking \
    --port 8188 \
    --tensor-parallel-size 1 \
    --quantization "wint8" \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --engine-worker-queue-port 8189 \
    --metrics-port 8190 \
    --cache-queue-port 8191 \
    --reasoning-parser ernie-45-vl-thinking \
    --tool-call-parser ernie-45-vl-thinking \
    --mm-processor-kwargs '{"image_max_pixels": 12845056}'
```
</details>

## Examples

### Running ERNIE-4.5-300B-A47B-Paddle

#### Start service

Deploy the ERNIE-4.5-300B-A47B-Paddle model with WINT4 precision and 32K context length on 4 XPUs

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # Specify which cards to be used
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**Note:** When deploying on 4 XPUs, only two configurations are supported which constrained by hardware limitations such as interconnect capabilities.
`export XPU_VISIBLE_DEVICES="0,1,2,3"`
or
`export XPU_VISIBLE_DEVICES="4,5,6,7"`

Refer to [Parameters](../parameters.md) for more options.

All supported models can be found in the *Supported Models* section above.

#### Send requests
Send requests using either curl or Python.

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Where is the capital of China?"}
  ]
}'
```

```python
import openai
host = "0.0.0.0"
port = "8188"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.completions.create(
    model="null",
    prompt="Where is the capital of China?",
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].text, end='')
print('\n')

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "Where is the capital of China?"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

For detailed OpenAI protocol specifications, see [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create). Differences from the standard OpenAI protocol are documented in [OpenAI Protocol-Compatible API Server](../online_serving/README.md).

### Running ERNIE-4.5-VL-28B-A3B-Paddle

#### Start service
Deploy the ERNIE-4.5-VL-28B-A3B-Paddle model with WINT8 precision and 32K context length on 1 XPU

```bash
export XPU_VISIBLE_DEVICES="0" # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle \
    --port 8188  \
    --tensor-parallel-size 1 \
    --quantization "wint8" \
    --max-model-len 32768 \
    --max-num-seqs 10 \
    --enable-mm \
    --mm-processor-kwargs '{"video_max_frames": 30}' \
    --limit-mm-per-prompt '{"image": 10, "video": 3}' \
    --reasoning-parser ernie-45-vl
```

#### Send requests

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg", "detail": "high"}},
              {"type": "text", "text": "Please describe the content of the image"}
            ]}
    ],
    "metadata": {"enable_thinking": false}
}'
```

```python
import openai

ip = "0.0.0.0"
service_http_port = "8188"
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg", "detail": "high"}},
              {"type": "text", "text": "Please describe the content of the image"}
            ]
        },
    ],
    temperature=0.0001,
    max_tokens=10000,
    stream=True,
    top_p=0,
    metadata={"enable_thinking": False},
)

def get_str(content_raw):
    content_str = str(content_raw) if content_raw is not None else ''
    return content_str

for chunk in response:
    if chunk.choices[0].delta is not None and chunk.choices[0].delta.role != 'assistant':
        reasoning_content = get_str(chunk.choices[0].delta.reasoning_content)
        content = get_str(chunk.choices[0].delta.content)
        print(reasoning_content + content, end='', flush=True)
print('\n')
```

### Running PaddleOCR-VL-0.9B

#### Start service

Deploy the PaddleOCR-VL-0.9B model with BF16 precision and 16K context length on 1 XPU

```bash
export FD_ENABLE_MAX_PREFILL=1
export XPU_VISIBLE_DEVICES="0" # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
   --model PaddlePaddle/PaddleOCR-VL \
   --port 8188 \
   --metrics-port 8181 \
   --engine-worker-queue-port 8182 \
   --max-model-len 16384 \
   --max-num-batched-tokens 16384 \
   --gpu-memory-utilization 0.8 \
   --max-num-seqs 256
```

#### Send requests

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/ocr_v5_eval/handwrite_ch_rec_val/中文手写古籍_000054_crop_32.jpg"}},
              {"type": "text", "text": "OCR:"}
            ]}
    ],
    "metadata": {"enable_thinking": false}
}'
```

```python
import openai

ip = "0.0.0.0"
service_http_port = "8188"
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/ocr_v5_eval/handwrite_ch_rec_val/中文手写古籍_000054_crop_32.jpg"}},
              {"type": "text", "text": "OCR:"}
            ]
        },
    ],
    temperature=0.0001,
    max_tokens=4096,
    stream=True,
    top_p=0,
    metadata={"enable_thinking": False},
)

def get_str(content_raw):
    content_str = str(content_raw) if content_raw is not None else ''
    return content_str

for chunk in response:
    if chunk.choices[0].delta is not None and chunk.choices[0].delta.role != 'assistant':
        reasoning_content = get_str(chunk.choices[0].delta.reasoning_content)
        content = get_str(chunk.choices[0].delta.content)
        print(reasoning_content + content, end='', flush=True)
print('\n')
```

### Running ERNIE-4.5-VL-28B-A3B-Thinking

#### Start service
Deploy the ERNIE-4.5-VL-28B-A3B-Thinking model with WINT8 precision and 128K context length on 1 XPU

```bash
export XPU_VISIBLE_DEVICES="0" # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking \
    --port 8188 \
    --tensor-parallel-size 1 \
    --quantization "wint8" \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --engine-worker-queue-port 8189 \
    --metrics-port 8190 \
    --cache-queue-port 8191 \
    --reasoning-parser ernie-45-vl-thinking \
    --tool-call-parser ernie-45-vl-thinking \
    --mm-processor-kwargs '{"image_max_pixels": 12845056 }'
```

#### Send requests

Initiate a service request through the following command
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Adapt Li Bai's "Silent Night Thoughts" into a modern poem"}
  ]
}'
```
When inputting images, initiate a request using the following command
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"Which era does the cultural relic in the picture belong to?"}
    ]}
  ]
}'
```
When inputting a video, initiate a request by following the following command
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"video_url", "video_url": {"url":"https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"}},
      {"type":"text", "text":"How many apples are there in the picture"}
    ]}
  ]
}'
```
When the input contains a tool call, initiate the request by following the command
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d $'{
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            },
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner, and the values of x1, y1, x2, y2 are all normalized to the range 0–1000 based on the original image dimensions."
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional)."
                        }
                    },
                    "required": [
                        "bbox_2d"
                    ]
                },
                "strict": false
            }
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is the old lady on the left side of the empty table behind older couple?"
                }
            ]
        }
    ],
    "stream": false
}'
```
When there are multiple requests and the tool returns results in the historical context, initiate the request by following the command below
When there are multiple requests and the tool returns results in the historical context, initiate the request by following the command below
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d $'{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Get the current weather in Beijing"
                }
            ]
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {
                            "location": "Beijing",
                            "unit": "c"
                        }
                    }
                }
            ],
            "content": ""
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": "location: Beijing，temperature: 23，weather: sunny，unit: c"
                }
            ]
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Determine weather in my location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "c",
                                "f"
                            ]
                        }
                    },
                    "additionalProperties": false,
                    "required": [
                        "location",
                        "unit"
                    ]
                },
                "strict": true
            }
        }
    ],
    "stream": false
}'
```
