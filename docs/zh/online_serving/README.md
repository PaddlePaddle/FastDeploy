# 兼容 OpenAI 协议的服务化部署

FastDeploy 提供与 OpenAI 协议兼容的服务化部署方案。用户可以通过如下命令快速进行部署：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768
```

如果要启用输出token的logprob，用户可以通过如下命令快速进行部署：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768 \
       --enable-logprob
```

服务部署时的命令行更多使用方式参考[参数说明](../parameters.md)。

## 发送用户请求

FastDeploy 接口兼容 OpenAI 协议，可以直接使用 OpenAI 的请求方式发送用户请求。

使用 curl 命令发送用户请求示例如下：

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

使用 curl 命令示例，演示如何在用户请求中包含logprobs参数：

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}, "logprobs": true, "top_logprobs": 5
  ]
}'
```

使用 Python 脚本发送用户请求示例如下：

```python
import openai
host = "0.0.0.0"
port = "8170"
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

关于 OpenAI 协议的说明可参考文档 [OpenAI Chat Compeltion API](https://platform.openai.com/docs/api-reference/chat/create)。

## 参数差异
### 请求参数差异
FastDeploy 与 OpenAI 协议的请求参数差异如下，其余请求参数会被忽略：
- `prompt` (仅支持 `v1/completions` 接口)
- `messages` (仅支持 `v1/chat/completions` 接口)
- `logprobs`: Optional[bool] = False (仅支持 `v1/chat/completions` 接口)
- `top_logprobs`: Optional[int] = None (仅支持 `v1/chat/completions` 接口。如果使用这个参数必须设置logprobs为True，取值大于等于0小于20)
- `frequency_penalty`: Optional[float] = 0.0
- `max_tokens`: Optional[int] = 16
- `presence_penalty`: Optional[float] = 0.0
- `stream`: Optional[bool] = False
- `stream_options`: Optional[StreamOptions] = None
- `temperature`: Optional[float] = None
- `top_p`: Optional[float] = None
- `metadata`: Optional[dict] = None (仅在v1/chat/compeltions中支持，用于配置额外参数, 如metadata={"enable_thinking": True})
  - `min_tokens`: Optional[int] = 1 最小生成的Token个数
  - `reasoning_max_tokens`: Optional[int] = None 思考内容最大Token数，默认与max_tokens一致
  - `enable_thinking`: Optional[bool] = True 支持深度思考的模型是否打开思考
  - `repetition_penalty`: Optional[float] = None: 直接对重复生成的token进行惩罚的系数（>1时惩罚重复，<1时鼓励重复）

> 注: 若为多模态模型 由于思考链默认打开导致输出过长，max tokens 可以设置为模型最长输出，或使用默认值。

### 返回字段差异

FastDeploy 增加的返回字段如下：

- `arrival_time`：返回所有 token 的累计耗时
- `reasoning_content`: 思考链的返回结果

返回参数总览：

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
