[简体中文](../zh/get_started/quick_start_qwen25_vl.md)

# Deploy Qwen2.5-VL in 10 Minutes

Before deployment, ensure your environment meets the following requirements:

- GPU Driver ≥ 535
- CUDA ≥ 12.3
- cuDNN ≥ 9.5
- Linux X86_64
- Python ≥ 3.10

This guide uses the lightweight Qwen2.5-VL model for demonstration, which can be deployed on most hardware configurations. Docker deployment is recommended.

For more information about how to install FastDeploy, refer to the [installation document](installation/README.md).

## 1. Launch Service

Please download the qwen25-vl model in advance: such as [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Add the following configuration in `config.json`
```text
  "rope_3d": true,
  "freq_allocation": 16
```

After installing FastDeploy, execute the following command in the terminal to start the service. For the configuration method of the startup command, refer to [Parameter Description](../parameters.md)

```
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model You/Path/Qwen2.5-VL-7B-Instruct \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

> 💡 Note: In the path specified by ```--model```, if the subdirectory corresponding to the path does not exist in the current directory, it will try to query whether AIStudio has a preset model based on the specified model name (such as ```Qwen/Qwen2.5-VL-7B-Instruct```). If it exists, it will automatically start downloading. The default download path is: ```~/xx```. For instructions and configuration on automatic model download, see [Model Download](../supported_models.md).
```--max-model-len``` indicates the maximum number of tokens supported by the currently deployed service.
```--max-num-seqs``` indicates the maximum number of concurrent processing supported by the currently deployed service.

**Related Documents**
- [Service Deployment](../online_serving/README.md)
- [Service Monitoring](../online_serving/metrics.md)

## 2. Request the Service
After starting the service, the following output indicates successful initialization:

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
