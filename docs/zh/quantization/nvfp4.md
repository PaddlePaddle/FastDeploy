[English](../../quantization/nvfp4.md)

# NVFP4量化
NVFP4 是 NVIDIA 引入的创新 4 位浮点格式，详细介绍请参考[Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。

基于[FlashInfer](https://github.com/flashinfer-ai/flashinfer), Fastdeploy 支持[Modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 产出格式的NVFP4量化模型推理。

- 注：目前该功能仅支持Ernie / Qwen系列的FP4量化模型。

## 如何使用
### 环境准备
#### 支持环境
- **支持硬件**：GPU sm >= 100
- **PaddlePaddle 版本**：3.3.0 或更高版本
- **Fastdeploy 版本**：2.5.0 或更高版本

#### Fastdeploy 安装
FastDeploy 需以 NVIDIA GPU 模式安装，具体安装方式请参考官方文档：[Fastdeploy NVIDIA GPU 环境安装指南](https://paddlepaddle.github.io/FastDeploy/zh/get_started/installation/nvidia_gpu/)。

### 运行推理服务

flashinfer-cutlass后端:
```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model nv-community/Qwen3-30B-A3B-FP4 \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --cache-queue-port 8183 \
    --tensor-parallel-size 1 \
    --max-model-len  32768 \
    --max-num-seqs 128
```

### flashinfer-cutedsl后端:

#### PaddlePaddle 兼容性补丁

由于 FlashInfer 与 PaddlePaddle 之间存在兼容性问题，需要在 `miniconda/envs/<your_env>/lib/python3.10/site-packages/` 中应用以下补丁：

1. **nvidia_cutlass_dsl/python_packages/cutlass/torch.py**

   将 `torch.device` 替换为 `"torch.device"`（作为字符串以避免冲突）。

2. **flashinfer/utils.py**

  修改 `get_compute_capability` 函数：
  ```bash
  @functools.cache
  def get_compute_capability(device: torch.device) -> Tuple[int, int]:
      return torch.cuda.get_device_capability(device)
      if device.type != "cuda":
          raise ValueError("device must be a cuda device")
      return torch.cuda.get_device_capability(device.index)
  ```

3. **flashinfer/cute_dsl/blockscaled_gemm.py**

   将 `cutlass_torch.current_stream()` 替换为：
   ```bash
   cuda.CUstream(torch.cuda.current_stream().stream_base.raw_stream)
   ```

### 运行推理服务

```bash
export FD_MOE_BACKEND="flashinfer-cutedsl"
export FD_USE_PFCC_DEEP_EP=1
export CUDA_VISIBLE_DEVICES=4,5,6,7



python -m fastdeploy.entrypoints.openai.multi_api_server \
       --ports "9811,9812,9813,9814" \
       --num-servers 4 \
       --model ERNIE-4.5-21B-A3B-FP4 \
       --disable-custom-all-reduce \
       --tensor-parallel-size 1 \
       --data-parallel-size 4 \
       --no-enable-prefix-caching \
       --max-model-len 65536 \
       --enable-expert-parallel \
       --num-gpu-blocks-override 8192 \
       --max-num-seqs 4 \
       --gpu-memory-utilization 0.9 \
       --max-num-batched-tokens 512 \
       --ep-prefill-use-worst-num-tokens \
       --graph-optimization-config '{"use_cudagraph":false}'
```

### 接口访问
通过如下命令发起服务请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```
```shell
curl -X POST "http://0.0.0.0:9811/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```

FastDeploy服务接口兼容OpenAI协议，可以通过如下Python代码发起服务请求。

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```
