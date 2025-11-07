# Tool_Calling

This document describes how to configure the server in FastDeploy to use the tool parsers, and how to invoke tools from the client.

FastDeploy GitHub: https://github.com/PaddlePaddle/FastDeploy

---
## Quickstart

### Starting FastDeploy with Tool Calling Enabled.

Start the server and enable tool call functionality. This example uses the ERNIE-4.5-21B-A3B model. We can use the ernie_x1 tool call parser in the fastdeploy directory, which supports parsing Hermes format tool call information:

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastdeploy.entrypoints.openai.api_server
    --model /models/ERNIE-4.5-21B-A3B \
    --port 8000 \
    --tool-call-parser ernie_x1 \
    --tool-parser-plugin FastDeploy/fastdeploy/entrypoints/openai/tool_parsers/ \
    --gpu-memory-utilization 0.9 \
    --load-choices "default_v1"
```
✅ FastDeploy does NOT require custom chat templates for tool use
Its tokenizer and chat template already support tool call formatting for Qwen, ERNIE, and compatible models.

### Client Example: Triggering Tool Calling
Make a request containing the tool to trigger the model to use the available tool:
```bash
#!/usr/bin/env python3
import json
import os
import httpx

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", 8566)
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

PAYLOAD = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are Qwen3-0.6B, a helpful assistant. "
                "Always decide whether you need to call a tool to answer the user. "
                "Only use the tools provided."
            ),
        },
        {"role": "user", "content": "What's the weather in BeiJing?"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, for example: Beijing",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["c", "f"],
                            "description": "Temperature units: c = Celsius, f = Fahrenheit",
                        },
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ],
    "stream": False,
}

def main():
    resp = httpx.post(URL, json=PAYLOAD, timeout=30000)
    resp.raise_for_status()
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```
Example output related to tool calls:
```text
...
        "tool_calls": [
          {
            "id": "chatcmpl-tool-bc90641c67e44dbfb981a79bc986fbe5",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"BeiJing\", \"unit\": \"c\"}"
            }
          }
        ],
...
```

## Parallel Tool Calls
If the model can generate parallel tool calls, FastDeploy will return a list:
```bash
tool_calls=[
  {"id": "...", "function": {...}},
  {"id": "...", "function": {...}}
]
```
FastDeploy passes through all calls in order, fully OpenAI compatible.

## Writing a Custom Tool Parser
FastDeploy currently uses a unified OpenAI tool-calling parser.If you want to create a custom plugin, FastDeploy exposes an entrypoint:[fastdeploy/entrypoints/openai/tool_parser/]
A custom parser should implement:
``` bash
class ToolParser:
    def extract(self, output_text: str) -> ToolCallResult:
        ...
```
Register with:
``` bash
ToolParserManager.register("my_parser", MyParser)
```
Enable via:
``` bash
--tool-parser-plugin <absolute path of the plugin file>
--tool-call-parser my_parser
```

---
