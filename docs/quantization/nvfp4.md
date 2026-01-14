
# NVFP4 Quantization
NVFP4 is an innovative 4-bit floating-point format introduced by NVIDIA. For detailed information, please refer to [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

Based on [FlashInfer](https://github.com/flashinfer-ai/flashinfer), Fastdeploy supports NVFP4 quantized model inference in the format produced by [Modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

- Note: Currently, this feature only supports FP4 quantized models of Ernie/Qwen series.

## How to Use
### Environment Setup
- **Supported Hardware**: GPU sm >= 100
- **PaddlePaddle Version**: 3.3.0 or higher
- **Fastdeploy Version**: 2.4.0 or higher

#### 1. Fastdeploy Installation
First, install the Fastdeploy base environment according to the [Fastdeploy NVIDIA GPU Environment Installation Guide](../../get_started/installation/nvidia_gpu.md).

#### 2. Flashinfer Installation
```bash
git clone -b support-paddlepaddle-with-compatible-api-and-tvmffi https://github.com/PFCCLab/flashinfer/ --recursive

cd flashinfer
python -m pip install -v .
```

### Running Inference Service
- Note: Need to set environment variable `export PADDLE_COMPATIBLE_API=true` and install the corresponding Flashinfer correctly
```bash
export PADDLE_COMPATIBLE_API=true
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
```.
