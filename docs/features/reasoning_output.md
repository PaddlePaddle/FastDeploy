[简体中文](../zh/features/reasoning_output.md)

# Reasoning Outputs

Reasoning models return an additional `reasoning_content` field in their output, which contains the reasoning steps that led to the final conclusion.

## Supported Models
| Model Name     | Parser Name    | Enable thinking by Default | Tool Calling  | Thinking switch  parameters|
|---------------|-------------|---------|---------|----------------|
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle  | ernie-45-vl | ✅       | ❌ | "chat_template_kwargs":{"enable_thinking": true/false}|
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | ernie-45-vl |    ✅    |  ❌  |"chat_template_kwargs":{"enable_thinking": true/false}|
| baidu/ERNIE-4.5-21B-A3B-Thinking  | ernie-x1  |   ✅ Not supported for turning off   | ✅|❌|
| baidu/ERNIE-4.5-VL-28B-A3B-Thinking  | ernie-45-vl-thinking  |   ✅ Not recommended to turn off   | ✅|"chat_template_kwargs": {"options": {"thinking_mode": "open/close"}}|

The reasoning model requires a specified parser to extract reasoning content. Referring to the `thinking switch parameters` of each model can turn off the model's thinking mode.

Interfaces that support toggling the reasoning mode:
1. `/v1/chat/completions` requests in OpenAI services.
2. `/v1/chat/completions` requests in the OpenAI Python client.
3. `llm.chat` requests in Offline interfaces.

For reasoning models, the length of the reasoning content can be controlled via `reasoning_max_tokens`. Add `"reasoning_max_tokens": 1024` to the request.

### Quick Start
When launching the model service, specify the parser name using the `--reasoning-parser` argument.
This parser will process the model's output and extract the `reasoning_content` field.

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --enable-mm \
    --tensor-parallel-size 8 \
    --port 8192 \
    --quantization wint4 \
    --reasoning-parser ernie-45-vl
```

Next, make a request to the model that should return the reasoning content in the response.
Taking the baidu/ERNIE-4.5-VL-28B-A3B-Paddle model as an example：

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
  "chat_template_kwargs":{"enable_thinking": true},
  "reasoning_max_tokens": 1024
}'
```

The `reasoning_content` field contains the reasoning steps to reach the final conclusion, while the `content` field holds the conclusion itself.

### Streaming chat completions
Streaming chat completions are also supported for reasoning models. The `reasoning_content` field is available in the `delta` field in `chat completion response chunks`

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
## Tool Calling
The reasoning content is also available when both tool calling and the reasoning parser are enabled. Additionally, tool calling only parses functions from the `content` field, not from the `reasoning_content`.

Model request example:
```bash
curl -X POST "http://0.0.0.0:8390/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "Get the current weather in BeiJing"
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
Model output example

```json
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "The user asks about ...",
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-311b9bda34274722afc654c55c8ce6a0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"BeiJing\", \"unit\": \"c\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ]
}
```
More reference documentation related to tool calling usage：  [Tool Calling](./tool_calling.md)
