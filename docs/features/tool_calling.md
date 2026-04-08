# Tool_Calling

This document describes how to configure the server in FastDeploy to use the tool parser, and how to invoke tools from the client.

## Tool Call parser for Ernie series models
| Model Name     | Parser Name       |
|---------------|-------------|
| baidu/ERNIE-4.5-21B-A3B-Thinking  | ernie-x1  |
| baidu/ERNIE-4.5-VL-28B-A3B-Thinking  | ernie-45-vl-thinking  |

## Quickstart

### Starting FastDeploy with Tool Calling Enabled.

Launch the server with tool-calling enabled.This example uses ERNIE-4.5-21B-A3B.Leverage the ernie-x1 reasoning parser and the ernie-x1 tool-call parser from the fastdeploy directory to extract the model’s reasoning content, response content, and the tool-calling information:

```bash
python -m fastdeploy.entrypoints.openai.api_server
    --model /models/ERNIE-4.5-21B-A3B \
    --port 8000 \
    --reasoning-parser ernie-x1 \
    --tool-call-parser ernie-x1
```
### Example of triggering tool calling
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
The example output is as follows. It shows that the model's output of the thought process `reasoning_content` and tool call information `tool_calls` was successfully parsed, and the current response content `content` is empty,`finish_reason` is `tool_calls`:
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
The parsed model output is as follows, containing the thought content `reasoning_content` and the response content `content`, with `finish_reason` set to stop:
```bash
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Today's weather in Beijing is sunny with a temperature of 23 degrees Celsius.",
                "reasoning_content": "User wants to ...",
                "tool_calls": null
            },
            "finish_reason": "stop"
        }
    ]
}
```
## Writing a Custom Tool Parser
FastDeploy supports custom tool parser plugins. You can refer to the following address to create a `tool parser`: `fastdeploy/entrypoints/openai/tool_parser`

A custom parser should implement:
``` python
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
Enable via:
``` bash
python -m fastdeploy.entrypoints.openai.api_server
--model <model path>
--tool-parser-plugin <absolute path of the plugin file>
--tool-call-parser my-parser
```

---
