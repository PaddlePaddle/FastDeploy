# 思考链内容

思考模型在输出中返回 `reasoning_content` 字段，表示思考链内容,即得出最终结论的思考步骤.

##目前支持思考链的模型
| 模型名称          | 解析器名称       | 默认开启思考链 |
|---------------|-------------|---------|
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle  | ernie-45-vl | ✓       |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | ernie-45-vl |    ✓    |

思考模型需要指定解析器,以便于对思考内容进行解析. 通过`enable_thinking=False` 参数可以关闭模型思考模式.

可以支持思考模式开关的接口:
1. OpenAI 服务中 `/v1/chat/completions`  请求.
2. OpenAI Python客户端中 `/v1/chat/completions`  请求.
3. Offline 接口中 `llm.chat`请求.

同时在思考模型中，支持通过```reasoning_max_tokens```控制思考内容的长度，在请求中添加```metadata={"reasoning_max_tokens": 1024}```即可。

## 快速使用
在启动模型服务时, 通过`--reasoning-parser`参数指定解析器名称.
该解析器会解析思考模型的输出, 提取`reasoning_content`字段.

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --enable-mm \
    --tensor-parallel-size 8 \
    --port 8192 \
    --quantization wint4 \
    --reasoning-parser ernie-45-vl
```

接下来, 向模型发送  `chat completion` 请求

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
  "metadata": {"enable_thinking": true}
}'

```

字段`reasoning_content`包含得出最终结论的思考步骤，而`content`字段包含最终结论。

### 流式会话
在流式会话中, `reasoning_content`字段会可以在`chat completion response chunks`中的 `delta` 中获取

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
    metadata={"enable_thinking": True}
)
for chunk in chat_response:
    if chunk.choices[0].delta is not None:
        print(chunk.choices[0].delta, end='')
        print("\n")

```
