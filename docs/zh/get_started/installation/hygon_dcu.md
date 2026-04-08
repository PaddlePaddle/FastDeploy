[English](../../../get_started/installation/hygon_dcu.md)

# 使用 FastDeploy 在海光 K100AI 上运行 ERNIE-4.5-300B-A47B & ERNIE-4.5-21B-A3B
当前版本软件只是作为K100AI + Fastdeploy 推理大模型的一个演示 demo，跑最新ERNIE4.5模型可能存在问题，后续进行修复和性能优化，给客户提供一个更稳定的版本。

## 准备机器
首先您需要准备以下配置的机器
- OS：Linux
- Python：3.10
- 内存：2T
- 磁盘：4T
- DCU 型号：K100AI
- DCU 驱动版本：≥ 6.3.8-V1.9.2

## 1. 使用 Docker 安装（推荐）

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

## 2. 启动服务

```bash
export FD_ATTENTION_BACKEND="BLOCK_ATTN"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "/models/ERNIE-45-Turbo/ERNIE-4.5-300B-A47B-Paddle/" \
    --port 8188 \
    --tensor-parallel-size 8 \
    --quantization=wint8 \
    --gpu-memory-utilization=0.8
```

### 请求服务

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
