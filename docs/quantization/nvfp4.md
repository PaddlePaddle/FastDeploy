[简体中文](../zh/quantization/nvfp4.md)

# NVFP4 Quantization
NVFP4 is an innovative 4-bit floating-point format introduced by NVIDIA. For detailed information, please refer to [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

Based on [FlashInfer](https://github.com/flashinfer-ai/flashinfer), Fastdeploy supports NVFP4 quantized model inference in the format produced by [Modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

- Note: Currently, this feature only supports FP4 quantized models of Ernie/Qwen series.

## How to Use
### Environment Setup
#### Supported Environment
- **Supported Hardware**: GPU sm >= 100
- **PaddlePaddle Version**: 3.3.0 or higher
- **Fastdeploy Version**: 2.5.0 or higher

#### FastDeploy Installation
Please ensure that FastDeploy is installed with NVIDIA GPU support.
Follow the official guide to set up the base environment: [Fastdeploy NVIDIA GPU Environment Installation Guide](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/).

### FlashInfer-cutedsl backend

#### PaddlePaddle Compatibility Patches for FlashInfer

Due to compatibility issues between FlashInfer and PaddlePaddle, you need to apply the following patches in `miniconda/envs/<your_env>/lib/python3.10/site-packages/`:

1. **nvidia_cutlass_dsl/python_packages/cutlass/torch.py**

   Replace `torch.device` with `"torch.device"` (as a string to avoid conflicts).

2. **flashinfer/utils.py**

  Modify the `get_compute_capability` function:
  ```bash
  @functools.cache
  def get_compute_capability(device: torch.device) -> Tuple[int, int]:
      return torch.cuda.get_device_capability(device)
      if device.type != "cuda":
          raise ValueError("device must be a cuda device")
      return torch.cuda.get_device_capability(device.index)
  ```

3. **flashinfer/cute_dsl/blockscaled_gemm.py**

  Replace `cutlass_torch.current_stream()` with:
  ```bash
  cuda.CUstream(torch.cuda.current_stream().stream_base.raw_stream)
  ```

#### Running Inference Service

flashinfer-cutlass backend:
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

flashinfer-cutedsl backend:
```bash
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

### API Access
Make service requests using the following command

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

FastDeploy service interface is compatible with OpenAI protocol. You can make service requests using the following Python code.

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
