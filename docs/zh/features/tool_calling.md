# Tool_Calling

本文档介绍如何在 FastDeploy 中配置服务器以使用工具解析器（tool parsers），以及如何在客户端调用工具。

FastDeploy GitHub: [https://github.com/PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy)

---

## 快速开始

### 启动包含解析器的FastDeploy

使用包含思考解析器和工具解析器的命令启动服务器。下面的示例使用 ERNIE-4.5-21B-A3B。我们可以使用 fastdeploy 目录中的 ernie-x1 思考解析器（thought parser）和 ernie-x1 工具调用解析器（tool-call parser）；从而实现解析模型的思考内容、回复内容以及工具调用信息：

```bash
python -m fastdeploy.entrypoints.openai.api_server
    --model /models/ERNIE-4.5-21B-A3B \
    --port 8000 \
    --reasoning-parser ernie-x1 \
    --tool-call-parser ernie-x1
```

### 客户端示例：触发工具调用

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

示例输出如下，可以看到成功解析出了模型输出的思考内容以及工具调用信息：
```bash
{
  "id": "chatcmpl-84d89bea-2e78-40ab-9618-d49983722140",
  "object": "chat.completion",
  "created": 1760098769,
  "model": "/root/paddlejob/models/Qwen3-30B-A3B-FP8",
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
        "prompt_token_ids": null,
        "completion_token_ids": null,
        "text_after_process": null,
        "raw_prediction": null,
        "prompt_tokens": null,
        "completion_tokens": null
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 242,
    "total_tokens": 363,
    "completion_tokens": 121,
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
```
## 并行工具调用（Parallel Tool Calls）

如果模型能够生成多个并行的工具调用，FastDeploy 会返回一个列表：

```bash
tool_calls=[
  {"id": "...", "function": {...}},
  {"id": "...", "function": {...}}
]
```

## 会话历史中包含工具调用的情况

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
        "location": "北京",
        "temperature": "23",
        "weather": "晴",
        "unit": "c"
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
解析出的模型输出结果如下，包含思考内容与回复内容：

```bash
{
    "id": "chatcmpl-6b881172-d927-4a6a-8113-3b0a9b257469",
    "object": "chat.completion",
    "created": 1754720554,
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "北京今天的天气是晴天，气温为23摄氏度。",
                "reasoning_content": "用户想...",
                "tool_calls": null
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 217,
        "total_tokens": 489,
        "completion_tokens": 272,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    }
}
```
## 编写自定义工具解析器
FastDeploy支持自定义工具解析器插件，可以在以下地址创建：`fastdeploy/entrypoints/openai/tool_parser/`

自定义解析器需要实现：

```python
@ToolParserManager.register_module("my-parser")
class ToolParser:
    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractToolCallInformation:
        ...
```

通过以下方式启用自定义解析器：

```bash
python -m fastdeploy.entrypoints.openai.api_server
--model <模型地址>
--tool-parser-plugin <自定义工具解释器的地址>
--tool-call-parser my-parser
```

---
