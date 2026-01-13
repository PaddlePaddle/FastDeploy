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
