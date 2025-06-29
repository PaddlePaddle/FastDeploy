# Chain-of-Thought Content

The reasoning model returns a `reasoning_content` field in the output, representing the chain-of-thought content—the reasoning steps that lead to the final conclusion.

## Currently Supported Chain-of-Thought Models
| Model Name     | Parser Name    | Chain-of-Thought Enabled by Default |
|----------------|----------------|-------------------------------------|
| ernie-45-vl    | ernie-45-vl    | ✓                                   |
| ernie-lite-vl  | ernie-45-vl    | ✓                                   |

The reasoning model requires a specified parser to interpret the reasoning content. The reasoning mode can be disabled by setting the `enable_thinking=False` parameter.

Interfaces that support toggling the reasoning mode:
1. `/v1/chat/completions` request in OpenAI services.
2. `/v1/chat/completions` request in the OpenAI Python client.
3. `llm.chat` request in Offline interfaces.

For reasoning models, the length of the reasoning content can be controlled via `reasoning_max_tokens`. Add `metadata={"reasoning_max_tokens": 1024}` to the request.

### Quick Start
When launching the model service, specify the parser name using the `--reasoning-parser` argument.  
This parser will process the model's output and extract the `reasoning_content` field.
```bash
python -m fastdeploy.entrypoints.openai.api_server --model /root/merge_llm_model  --enable-mm --tensor-parallel-size=8 --port 8192 --quantization wint4 --reasoning-parser=ernie-45-vl
```

Next, send a `chat completion` request to the model:
```bash
curl -X POST "http://0.0.0.0:8192/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "Which era does the cultural relic in the picture belong to"}
    ]}
  ],
  "metadata": {"enable_thinking": true}
}'
```
The `reasoning_content` field contains the reasoning steps to reach the final conclusion, while the `content` field holds the conclusion itself.

### Streaming Sessions
In streaming sessions, the `reasoning_content` field can be retrieved from the `delta` in `chat completion response chunks`.
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
        {"type": "text", "text": "Which era does the cultural relic in the picture belong to"}]}
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