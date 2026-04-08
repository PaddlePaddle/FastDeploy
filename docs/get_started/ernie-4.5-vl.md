[简体中文](../zh/get_started/ernie-4.5-vl.md)

# Deploy ERNIE-4.5-VL-424B-A47B Multimodal Model

This document explains how to deploy the ERNIE-4.5-VL multimodal model, which supports users to interact with the model using multimodal data (including reasoning capabilities). Before starting the deployment, please ensure that your hardware environment meets the following requirements:

- GPU Driver >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 8 GPUs

Refer to the [Installation Guide](./installation/README.md) for FastDeploy setup.

>💡 **Note**: ERNIE multimodal models all support thinking mode, which can be enabled by setting ```enable_thinking``` when initiating a service request (see the example below)..

## Prepare the Model
Specify ```--model baidu/ERNIE-4.5-VL-424B-A47B-Paddle``` during deployment to automatically download the model from AIStudio with resumable downloads. You can also manually download the model from other sources. Note that FastDeploy requires Paddle-format models. For more details, see [Supported Models](../supported_models.md).

## Launch the Service

Execute the following command to start the service. For parameter configurations, refer to [Parameter Guide](../parameters.md).

>💡 **Note**: Since the model parameter size is 424B-A47B, on an 80G * 8 GPU machine, specify ```--quantization wint4``` (wint8 is also supported).

```shell
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-424B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8183 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --mm-processor-kwargs '{"video_max_frames": 30}' \
       --limit-mm-per-prompt '{"image": 10, "video": 3}' \
       --reasoning-parser ernie-45-vl
```

## Request the Service
After launching, the service is ready when the following logs appear:

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

### Health Check

Verify service status (HTTP 200 indicates success):

```shell
curl -i http://0.0.0.0:8180/health
```

### cURL Request
Send requests as follows:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Rewrite Li Bai's 'Quiet Night Thoughts' as a modern poem"}
  ]
}'
```
\
For image inputs:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"From which era does the artifact in the image originate?"}
    ]}
  ]
}'
```
Image can also be provided through base64-encoded string:
```shell
{"type":"image_url", "image_url": {"url":"data:image/jpg;base64,this/is/an/example"}
```
or absolute path to local file:
```shell
{"type":"image_url", "image_url": {"url":"file:///this/is/an/example"}
```
\
For video inputs:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"video_url", "video_url": {"url":"https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"}},
      {"type":"text", "text":"How many apples are in the scene?"}
    ]}
  ]
}'
```
Video can also be provided through base64-encoded string:
```shell
{"type":"video_url", "video_url": {"url":"data:video/mp4;base64,this/is/an/example"}
```
or absolute path to local file:
```shell
{"type":"video_url", "video_url": {"url":"file:///this/is/an/example"}
```
\
ERNIE-4.5-VL supports reasoning mode (enabled by default). Disable it as follows:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "From which era does the artifact in the image originate?"}
    ]}
  ],
  "chat_template_kwargs":{"enable_thinking": false}
}'
```

### Python Client (OpenAI-compatible API)

FastDeploy's API is OpenAI-compatible. You can also use Python for streaming requests:

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
            {"type": "text", "text": "From which era does the artifact in the image originate?"},
        ]},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## Model Output
Example output with reasoning (reasoning content in `reasoning_content`, response in `content`):

```json
{
    "id": "chatcmpl-c4772bea-1950-4bf4-b5f8-3d3c044aab06",
    "object": "chat.completion",
    "created": 1750236617,
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The artifact in the image ...",
                "reasoning_content": "The user asks about ..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1260,
        "total_tokens": 2042,
        "completion_tokens": 782
    }
}
```

Example output without reasoning:

```json
{
    "id": "chatcmpl-4d508b96-0ea1-4430-98a6-ae569f74f25b",
    "object": "chat.completion",
    "created": 1750236495,
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The artifact is a ...",
                "reasoning_content": null
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1265,
        "total_tokens": 1407,
        "completion_tokens": 142
    }
}
```
