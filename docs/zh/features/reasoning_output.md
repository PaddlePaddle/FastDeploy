[English](../../features/reasoning_output.md)

# 思考链内容

思考模型在输出中返回 `reasoning_content` 字段，表示思考链内容,即得出最终结论的思考步骤.

## 目前支持思考链的模型
| 模型名称          | 解析器名称       | 默认开启思考链 | 工具调用  | 思考开关控制参数|
|---------------|-------------|---------|---------|--------- |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle  | ernie-45-vl | ✅       | ❌ |  "chat_template_kwargs":{"enable_thinking": true/false}|
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | ernie-45-vl |    ✅    |  ❌  |"chat_template_kwargs":{"enable_thinking": true/false}|
| baidu/ERNIE-4.5-21B-A3B-Thinking  | ernie-x1  |   ✅不支持关思考  | ✅|❌|
| baidu/ERNIE-4.5-VL-28B-A3B-Thinking  | ernie-45-vl-thinking  |   ✅不推荐关闭   | ✅|"chat_template_kwargs": {"options": {"thinking_mode": "open/close"}}|

思考模型需要指定解析器,以便于对思考内容进行解析. 参考各个模型的 `思考开关控制参数` 可以关闭模型思考模式.

可以支持思考模式开关的接口:
1. OpenAI 服务中 `/v1/chat/completions`  请求.
2. OpenAI Python客户端中 `/v1/chat/completions`  请求.
3. Offline 接口中 `llm.chat`请求.

同时在思考模型中，支持通过 `reasoning_max_tokens` 控制思考内容的长度，在请求中添加 `"reasoning_max_tokens": 1024` 即可。

## 快速使用
在启动模型服务时, 通过 `--reasoning-parser` 参数指定解析器名称.
该解析器会解析思考模型的输出, 提取 `reasoning_content` 字段.

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --enable-mm \
    --tensor-parallel-size 8 \
    --port 8192 \
    --quantization wint4 \
    --reasoning-parser ernie-45-vl
```

接下来, 向模型发送  `chat completion` 请求， 以`baidu/ERNIE-4.5-VL-28B-A3B-Paddle`模型为例

```bash
curl -X POST "http://0.0.0.0:8192/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "图中的文物属于哪个年代"}
    ]}
  ],
  "chat_template_kwargs":{"enable_thinking": true},
  "reasoning_max_tokens": 1024
}'

```

字段 `reasoning_content` 包含得出最终结论的思考步骤，而 `content` 字段包含最终结论。

### 流式会话
在流式会话中, `reasoning_content` 字段会可以在 `chat completion response chunks` 中的 `delta` 中获取

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8192/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
chat_response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": [ {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
        {"type": "text", "text": "图中的文物属于哪个年代"}]}
    ],
    model="vl",
    stream=True,
    extra_body={
      "chat_template_kwargs":{"enable_thinking": True},
      "reasoning_max_tokens": 1024
    }
)
for chunk in chat_response:
    if chunk.choices[0].delta is not None:
        print(chunk.choices[0].delta, end='')
        print("\n")

```
## 工具调用
如果模型支持工具调用， 可以同时启动模型回复内容的思考链解析  `reasoning_content` 及工具解析 `tool-call-parser`。 工具内容仅从模型回复内容 `content` 中进行解析，而不会影响思考链内容。
例如，
```bash
curl -X POST "http://0.0.0.0:8390/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "北京今天天气怎么样？"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Determine weather in my location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": [
                "c",
                "f"
              ]
            }
          },
          "additionalProperties": false,
          "required": [
            "location",
            "unit"
          ]
        },
        "strict": true
      }
    }],
    "stream": false
}'
```
返回结果示例如下：
```json
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "用户问的是..",
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-311b9bda34274722afc654c55c8ce6a0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"北京\", \"unit\": \"c\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ]
}
```
更多工具调用相关的使用参考文档  [Tool Calling](./tool_calling.md)
