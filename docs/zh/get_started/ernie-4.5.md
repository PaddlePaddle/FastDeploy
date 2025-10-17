[English](../../get_started/ernie-4.5.md)

# ERNIE-4.5模型

本文档讲解如何部署ERNIE-4.5模型，在开始部署前，请确保你的硬件环境满足如下条件：

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 4卡

安装FastDeploy方式参考[安装文档](./installation/README.md)。

## 准备模型
部署时指定 ```--model baidu/ERNIE-4.5-300B-A47B-Paddle``` 即可自动从AIStudio下载模型，并支持断点续传。你也可以自行从不同渠道下载模型，需要注意的是FastDeploy依赖Paddle格式的模型，更多说明参考[支持模型列表](../supported_models.md)。

## 启动服务

>💡 **注意**： 由于模型参数量为300B-A47B，在80G * 8卡的机器上，需指定 ```--quantization wint4``` (wint8也可部署，其中wint4 4卡即可部署，wint8则需要8卡)。

执行如下命令，启动服务，其中启动命令配置方式参考[参数说明](../parameters.md)。

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

## 用户发起服务请求
执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

FastDeploy提供服务探活接口，用以判断服务的启动状态，执行如下命令返回 ```HTTP/1.1 200 OK``` 即表示服务启动成功。

```shell
curl -i http://0.0.0.0:8180/health
```

通过如下命令进行服务请求。

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
