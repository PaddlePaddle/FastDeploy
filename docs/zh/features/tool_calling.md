# Tool_Calling

本文档介绍如何在 FastDeploy 中配置服务器以使用工具解析器（tool parser），以及如何在客户端调用工具。

## Ernie系列模型配套工具解释器
| 模型名称          | 解析器名称       |
|---------------|-------------|
| baidu/ERNIE-4.5-21B-A3B-Thinking  | ernie-x1  |
| baidu/ERNIE-4.5-VL-28B-A3B-Thinking  | ernie-45-vl-thinking  |

## 快速开始

### 启动包含解析器的FastDeploy

使用包含思考解析器和工具解析器的命令启动服务器。下面的示例使用 ERNIE-4.5-21B-A3B。我们可以使用 fastdeploy 目录中的 ernie-x1 思考解析器（reasoning parser）和 ernie-x1 工具调用解析器（tool-call parser）；从而实现解析模型的思考内容、回复内容以及工具调用信息：

```bash
python -m fastdeploy.entrypoints.openai.api_server
    --model /models/ERNIE-4.5-21B-A3B \
    --port 8000 \
    --reasoning-parser ernie-x1 \
    --tool-call-parser ernie-x1
```

### 触发工具调用示例

构造一个包含工具的请求以触发模型调用工具：

```python
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
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
          "description": "获取指定地点的当前天气",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "城市名，如：北京。"
              },
              "unit": {
                "type": "string",
                "enum": ["c", "f"],
                "description": "温度单位：c = 摄氏度，f = 华氏度"
              }
            },
            "required": ["location", "unit"],
            "additionalProperties": false
          },
          "strict": true
        }
      }
    ]
  }'
```

示例输出如下，可以看到成功解析出了模型输出的思考内容`reasoning_content`以及工具调用信息`tool_calls`，且当前的回复内容`content`为空,`finish_reason`为工具调用`tool_calls`：
```bash
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "multimodal_content": null,
                "reasoning_content": "User wants to ... ",
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-bc90641c67e44dbfb981a79bc986fbe5",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"北京\", \"unit\": \"c\"}"
                        }
                    }
                ],
                "finish_reason": "tool_calls"
            }
        }
    ]
}
```
## 并行工具调用

如果模型能够生成多个并行的工具调用，FastDeploy 会返回一个列表：

```bash
tool_calls=[
  {"id": "...", "function": {...}},
  {"id": "...", "function": {...}}
]
```

## 工具调用结果出现在历史会话中

如果前几轮对话中包含工具调用，可以按以下方式构造请求：

```python
curl -X POST "http://0.0.0.0:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "你好，北京天气怎么样？"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "北京",
              "unit": "c"
            }
          }
        }
      ],
      "thoughts": "用户需要查询北京今天的天气。"
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": {
        "type": "text",
        "text": "{\"location\": \"北京\",\"temperature\": \"23\",\"weather\": \"晴\",\"unit\": \"c\"}"
        }
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定位置的当前天气。",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "城市名称，例如：北京"
            },
            "unit": {
              "type": "string",
              "enum": [
                "c",
                "f"
              ],
              "description": "温度单位：c = 摄氏度，f = 华氏度"
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
    }
  ]
}'
```
解析出的模型输出结果如下，包含思考内容`reasoning_content`与回复内容`content`，且`finish_reason`为`stop`：
```bash
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "北京今天的天气是晴天，气温为23摄氏度。",
                "reasoning_content": "用户想...",
                "tool_calls": null
            },
            "finish_reason": "stop"
        }
    ]
}
```
## 编写自定义工具解析器
FastDeploy支持自定义工具解析器插件，可以参考以下地址中的`tool parser`创建：`fastdeploy/entrypoints/openai/tool_parser`

自定义解析器需要实现：

```python
# import the required packages
# register the tool parser to ToolParserManager
@ToolParserManager.register_module("my-parser")
class ToolParser:
    def __init__(self, tokenizer: AnyTokenizer):
      super().__init__(tokenizer)

    # implement the tool parse for non-stream call
    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractToolCallInformation:
      return ExtractedToolCallInformation(tools_called=False,tool_calls=[],content=text)

    # implement the tool call parse for stream call
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return delta
```

通过以下方式启用自定义解析器：

```bash
python -m fastdeploy.entrypoints.openai.api_server
--model <模型地址>
--tool-parser-plugin <自定义工具解释器的地址>
--tool-call-parser my-parser
```

---
