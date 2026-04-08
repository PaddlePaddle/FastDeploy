[简体中文](../zh/get_started/ernie-4.5.md)

# Deploy ERNIE-4.5-300B-A47B Model

This document explains how to deploy the ERNIE-4.5 model. Before starting the deployment, please ensure that your hardware environment meets the following requirements:

- GPU Driver >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 4 GPUs

For FastDeploy installation, refer to the [Installation Guide](./installation/README.md).

## Prepare the Model
Specify `--model baidu/ERNIE-4.5-300B-A47B-Paddle` during deployment to automatically download the model from AIStudio with support for resumable transfers. Alternatively, you can download the model manually from other sources. Note that FastDeploy requires the model in Paddle format. For more details, see the [Supported Models List](../supported_models.md).

## Start the Service

>💡 **Note**: Since the model parameter size is 300B-A47B,, on an 80G * 8-GPU machine, specify `--quantization wint4` (wint8 is also supported, where wint4 requires 4 GPUs and wint8 requires 8 GPUs).

Execute the following command to start the service. For configuration details, refer to the [Parameter Guide](../parameters.md):

```shell
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8183 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

## Request the Service
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
Send requests to the service with the following command:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Write me a poem about large language model."}
  ]
}'
```

### Python Client (OpenAI-compatible API)

FastDeploy's API is OpenAI-compatible. You can also use Python for requests:

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Write me a poem about large language model."},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```
