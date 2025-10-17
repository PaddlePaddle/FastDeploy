[简体中文](../../zh/get_started/installation/hygon_dcu.md)

# Run ERNIE-4.5-300B-A47B & ERNIE-4.5-21B-A3B model on hygon machine
The current version of the software merely serves as a demonstration demo for the hygon k100AI combined with the Fastdeploy inference framework for large models. There may be issues when running the latest ERNIE4.5 model, and we will conduct repairs and performance optimization in the future. Subsequent versions will provide customers with a more stable version.

## Requirements
Firstly, you need to prepare a machine with the following configuration
- OS：Linux
- Python：3.10
- Memory: 2T
- Disk: 4T
- DCU Model：K100AI
- DCU Driver Version：≥ 6.3.8-V1.9.2

## 1. Set up using Docker (Recommended)

```bash
mkdir Work
cd Work
docker pull image.sourcefind.cn:5000/dcu/admin/base/custom:fastdeploy2.0.0-kylinv10-dtk25.04-py3.10

docker run -it \
--network=host \
--name=ernie45t \
--privileged \
--device=/dev/kfd \
--device=/dev/dri \
--ipc=host \
--shm-size=16G \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-u root \
--ulimit stack=-1:-1 \
--ulimit memlock=-1:-1 \
-v `pwd`:/home \
-v /opt/hyhal:/opt/hyhal:ro \
image.sourcefind.cn:5000/dcu/admin/base/custom:fastdeploy2.0.0-kylinv10-dtk25.04-py3.10 /bin/bash
```

## 2. Start service

```bash
export FD_ATTENTION_BACKEND="BLOCK_ATTN"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "/models/ERNIE-45-Turbo/ERNIE-4.5-300B-A47B-Paddle/" \
    --port 8188 \
    --tensor-parallel-size 8 \
    --quantization=wint8 \
    --gpu-memory-utilization=0.8
```

### Send requests

Send requests using either curl or Python

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

ip = "0.0.0.0"
service_http_port = "8188"
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?"},
    ],
    temperature=1,
    max_tokens=1024,
    stream=False,
)
print(response)
```
