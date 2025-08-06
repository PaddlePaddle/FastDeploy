# 10分钟完成 ERNIE-4.5-VL-28B-A3B-Paddle 多模态模型部署

本文档讲解如何部署ERNIE-4.5-VL-28B-A3B-Paddle模型，在开始部署前，请确保你的硬件环境满足如下条件：

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 运行模型满足最低硬件配置要求，参考[支持模型列表文档](../supported_models.md)

为了快速在各类硬件部署，本文档采用 ```ERNIE-4.5-VL-28B-A3B-Paddle``` 多模态模型作为示例，可在大部分硬件上完成部署。

安装FastDeploy方式参考[安装文档](./installation/README.md)。

>💡 **提示**： ERNIE多模态系列模型均支持思考模式，可以通过在发起服务请求时设置 ```enable_thinking``` 开启（参考如下示例）。

## 1. 启动服务
安装FastDeploy后，在终端执行如下命令，启动服务，其中启动命令配置方式参考[参数说明](../parameters.md)

```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-28B-A3B-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --reasoning-parser ernie-45-vl
```

>💡 注意：在 ```--model``` 指定的路径中，若当前目录下不存在该路径对应的子目录，则会尝试根据指定的模型名称（如 ```baidu/ERNIE-4.5-0.3B-Base-Paddle```）查询AIStudio是否存在预置模型，若存在，则自动启动下载。默认的下载路径为：```~/xx```。关于模型自动下载的说明和配置参阅[模型下载](../supported_models.md)。
```--max-model-len``` 表示当前部署的服务所支持的最长Token数量。
```--max-num-seqs``` 表示当前部署的服务所支持的最大并发处理数量。
```--reasoning-parser``` 指定思考内容解析器。
```--enable-mm``` 表示是否开启多模态支持。

**相关文档**

- [服务部署配置](../online_serving/README.md)
- [服务监控metrics](../online_serving/metrics.md)

## 2. 用户发起服务请求

执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```
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

通过如下命令发起服务请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "图中的文物属于哪个年代"}
    ]}
  ],
  "chat_template_kwargs":{"enable_thinking": false}
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
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
            {"type": "text", "text": "图中的文物属于哪个年代?"},
        ]},
    ],
    extra_body={"enable_thinking": false},
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```
