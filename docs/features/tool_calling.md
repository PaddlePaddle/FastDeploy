# Tool_Calling

This document describes how to configure the server in FastDeploy to use the tool parsers, and how to invoke tools from the client.

FastDeploy GitHub: https://github.com/PaddlePaddle/FastDeploy

---
## Quickstart

### Starting FastDeploy with Tool Calling Enabled.

Launch the server with tool-calling enabled.This example uses ERNIE-4.5-21B-A3B.Leverage the ernie-x1 thought parser and the ernie-x1 tool-call parser from the fastdeploy directory to extract the model’s reasoning block, final answer, and any tool-calling information:

```bash
python -m fastdeploy.entrypoints.openai.api_server
    --model /models/ERNIE-4.5-21B-A3B \
    --port 8000 \
    --reasoning-parser ernie-x1 \
    --tool-call-parser ernie-x1
```
### Client Example: Triggering Tool Calling
Make a request containing the tool to trigger the model to use the available tool:
```python
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What's the weather in Beijing?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "City name, for example: Beijing"
              },
              "unit": {
                "type": "string",
                "enum": ["c", "f"],
                "description": "Temperature units: c = Celsius, f = Fahrenheit"
              }
            },
            "required": ["location", "unit"],
            "additionalProperties": false
          },
          "strict": true
        }
      }
    ],
    "stream": false
  }'
```
The output for this sample request is shown below; note that both the reasoning block and the tool-call information emitted by the model have been successfully parsed:
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
              "arguments": "{\"location\": \"Beijing\", \"unit\": \"c\"}"
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
  ]
  }
}
```

## Parallel Tool Calls
If the model can generate parallel tool calls, FastDeploy will return a list:
```bash
tool_calls=[
  {"id": "...", "function": {...}},
  {"id": "...", "function": {...}}
]
```

## Requests containing tools in the conversation history
If tool-call information exists in previous turns, you can construct the request as follows:
```python
curl -X POST "http://0.0.0.0:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "Hello,What's the weather in Beijing?"
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
              "location": "Beijing",
              "unit": "c"
            }
          }
        }
      ],
      "thoughts": "Users need to check today's weather in Beijing."
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": {
        "location": "Beijing",
        "temperature": "23",
        "weather": "sunny",
        "unit": "c"
      }
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
    }
  ],
  "stream": false
}'
```
The parsed model output that can be obtained is as follows, comprising both the reasoning content and the response content:
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
                "content": "Today's weather in Beijing is sunny with a temperature of 23 degrees Celsius.",
                "reasoning_content": "User wants to ...",
                "tool_calls": null
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ]
}
```
## Writing a Custom Tool Parser
FastDeploy supports custom tool-call parser plug-ins; you can create one under:`fastdeploy/entrypoints/openai/tool_parser/`

A custom parser should implement:
``` python
@ToolParserManager.register_module("my-parser")
class ToolParser:
    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractToolCallInformation:
        ...
```
Enable via:
``` bash
python -m fastdeploy.entrypoints.openai.api_server
--model <model path>
--tool-parser-plugin <absolute path of the plugin file>
--tool-call-parser my-parser
```

---
