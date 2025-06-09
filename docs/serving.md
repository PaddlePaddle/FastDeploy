# 服务化部署

使用如下命令进行服务部署

```bash
python -m fastdeploy.entrypoints.openai.api_server --model ernie-45-turbo --port 8188 --tensor-parallel-size 8
```

其中api_server支持的参数包括

* --host: 服务配置的hostname

* --port: 服务配置的HTTP端口

* --metrics-port: 服务配置的metrics端口 详细参考[metrics说明](./metrics.md)

* --workers: api-server基于uvicorn启动时的进程数

其余参数为引擎配置，可直接参考[离线推理](./offline_inference.md)中fastdeploy.LLM的参数说明，

* --model

* --max-model-len

* --block-size

* --use-warmup

* --engine-worker-queue-port

* --tensor-parallel-size

* --max-num-seqs

* --num-gpu-blocks-override

* --max-num-batched-tokens

* --gpu-memory-utilization

* --kv-cache-ratio

* --enable-mm

除上述参数外，服务在启动时同步也包含Scheduler(包含LocalScheduler单实例服务或GlobalScheduler多实例负载均衡),相关使用说明可参考[Scheduler文档)(./scheduler.md)。

## 请求服务

FastDeploy服务接口兼容OpenAI协议，因此可以直接使用openai的请求方式请求服务，如下分别提供curl和python示例,

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

```python
import openai

ip = "0.0.0.0"
service_http_port = "8188"    # 服务配置的
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

# 非流式对话
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},   # system不是必需，可选
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=1,
    max_tokens=1024,
    stream=False,
)
print(response)

# 流式对话，历史多轮
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},   # system不是必需，可选
        {"role": "user", "content": "List 3 countries and their capitals."},
        {"role": "assistant", "content": "China(Beijing), France(Paris), Australia(Canberra)."},
        {"role": "user", "content": "OK, tell more."},
    ],
    temperature=1,
    max_tokens=1024,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta is not None:
        print(chunk.choices[0].delta, end='')
        print("\n")
```

关于OpenAI协议的说明可参考文档 OpenAI Chat Compeltion API，需要说明的是，FastDeploy提供的服务在参数上存在如下差异，

1. 仅支持OpenAI如下参数（其余参数配置会被服务忽略）
- prompt (v1/completions)
- messages(v1/chat/completions)
- frequency_penalty: Optional[float] = 0.0
- max_tokens: Optional[int] = 16
- presence_penalty: Optional[float] = 0.0
- seed: Optional[int] = None
- stream: Optional[bool] = False
- stream_options: Optional[StreamOptions] = None
- temperature: Optional[float] = None
- top_p: Optional[float] = None
- metadata: Optional[dict] = None (仅在v1/chat/compeltions中支持，用于配置min_tokens，例如metadata={"min_tokens": 20})

> 注:若为X1 模型 由于思考链默认打卡导致输出过长，max tokens 可以设置为模型最长输出，或无需设置

2. 在返回的信息

新增返回参数：
arrival_time ：每个token 的返回的累计耗时
reasoning_content: 思考链返回结果

```python
ChatCompletionStreamResponse:
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
 ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    arrival_time: Optional[float] = None
DeltaMessage:
    role: Optional[str] = None
    content: Optional[str] = None
    token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None
```
