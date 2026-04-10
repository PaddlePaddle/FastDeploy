[English](../../usage/kunlunxin_xpu_deployment.md)

## 支持的模型

注：以下模型支持和部署命令仅适用于 2.5.0 版本
<details>
<summary><b>ERNIE-4.5-300B-A47B (32K, WINT8, 8 卡)</b> </summary>

**快速启动：**
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

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4  # 通过 `xpu-smi topo -m` 命令查看机器的RDMA网卡名称
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
<summary><b>ERNIE-4.5-300B-A47B (32K, WINT4, 4 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3"  # 或 "4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3"  # 或 "4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2  # 通过 `xpu-smi topo -m` 命令查看机器的RDMA网卡名称
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
<summary><b>ERNIE-4.5-300B-A47B (32K, WINT4, 8 卡)</b> </summary>

**快速启动：**
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

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4  # 通过 `xpu-smi topo -m` 命令查看机器的RDMA网卡名称
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
<summary><b>ERNIE-4.5-300B-A47B (128K, WINT4, 8 卡)</b> </summary>

**快速启动：**
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

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4 # 通过 `xpu-smi topo -m` 命令查看机器的RDMA网卡名称
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
<summary><b>ERNIE-4.5-21B-A3B (32K, BF16, 1 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-21B-A3B (32K, WINT8, 1 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-21B-A3B (32K, WINT4, 1 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-21B-A3B (128K, BF16, 1 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-21B-A3B (128K, WINT8, 1 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint8" \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-21B-A3B (128K, WINT4, 1 卡)</b> </summary>

**快速启动：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**性能更优：**
```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-0.3B (32K, BF16, 1 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-0.3B (32K, WINT8, 1 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-0.3B (128K, BF16, 1 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-0.3B (128K, WINT8, 1 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-300B-A47B-W4A8C8-TP4 (32K, W4A8, 4 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3"  # 或 "4,5,6,7"
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
<summary><b>ERNIE-4.5-VL-28B-A3B (32K, WINT8, 1 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-VL-424B-A47B (32K, WINT8, 8 卡)</b> </summary>

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
<summary><b>PaddleOCR-VL-0.9B (32K, BF16, 1 卡)</b> </summary>

```bash
export FD_ENABLE_MAX_PREFILL=1
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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
<summary><b>ERNIE-4.5-VL-28B-A3B-Thinking (128K, WINT8, 1 卡)</b> </summary>

```bash
export XPU_VISIBLE_DEVICES="0"  # 指定任意一张卡
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

## 示例

### 运行ERNIE-4.5-300B-A47B-Paddle
#### 启动服务

基于 WINT4 精度和 32K 上下文部署 ERNIE-4.5-300B-A47B-Paddle 模型到 4 卡 P800 服务器

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # 设置使用的 XPU 卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**注意：** 使用 P800 在 4 块 XPU 上进行部署时，由于受到卡间互联拓扑等硬件限制，仅支持以下两种配置方式：
`export XPU_VISIBLE_DEVICES="0,1,2,3"`
or
`export XPU_VISIBLE_DEVICES="4,5,6,7"`

更多参数可以参考 [参数说明](../parameters.md)。

全部支持的模型可以在上方的 *支持的模型* 章节找到。

#### 请求服务

您可以基于 OpenAI 协议，通过 curl 和 python 两种方式请求服务。

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

OpenAI 协议的更多说明可参考文档 [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create)，以及与 OpenAI 协议的区别可以参考 [兼容 OpenAI 协议的服务化部署](../online_serving/README.md)。

### 运行ERNIE-4.5-VL-28B-A3B-Paddle

#### 启动服务

基于 WINT8 精度和 32K 上下文部署 ERNIE-4.5-VL-28B-A3B-Paddle 模型到 单卡 P800 服务器

```bash
export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡
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

#### 请求服务

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg", "detail": "high"}},
              {"type": "text", "text": "请描述图片内容"}
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
              {"type": "text", "text": "请描述图片内容"}
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

### 运行PaddleOCR-VL-0.9B

#### 启动服务

基于 BF16 精度和 16K 上下文部署 PaddleOCR-VL-0.9B 模型到 单卡 P800 服务器

```bash
export FD_ENABLE_MAX_PREFILL=1
export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡
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

#### 请求服务

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo.jpg"}},
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
              {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo.jpg"}},
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

### 运行ERNIE-4.5-VL-28B-A3B-Thinking

#### 启动服务

基于 WINT8 精度和 128K 上下文部署 ERNIE-4.5-VL-28B-A3B-Thinking 模型到 单卡 P800 服务器

```bash
export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡
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

### 请求服务
通过如下命令发起服务请求
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```
输入包含图片时，按如下命令发起请求
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"图中的文物属于哪个年代?"}
    ]}
  ]
}'
```
输入包含视频时，按如下命令发起请求
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"video_url", "video_url": {"url":"https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"}},
      {"type":"text", "text":"画面中有几个苹果?"}
    ]}
  ]
}'
```
输入包含工具调用时，按如下命令发起请求
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
多轮请求， 历史上下文中包含工具返回结果时，按如下命令发起请求
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
