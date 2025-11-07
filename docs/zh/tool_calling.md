# Tool_Calling

本文描述如何在 FastDeploy 中配置服务器以使用工具解析器，以及如何在客户端触发工具调用。

FastDeploy GitHub: [https://github.com/PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy)

---

## 快速开始

### 启动 FastDeploy，并开启 Tool Calling 功能

启动服务并启用工具调用功能。下面示例使用 ERNIE-4.5-21B-A3B 模型。我们可以使用 fastdeploy 目录下的 `ernie_x1` 工具调用解析器，它支持解析 Hermes 格式的工具调用信息：

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastdeploy.entrypoints.openai.api_server
    --model /models/ERNIE-4.5-21B-A3B \
    --port 8000 \
    --tool-call-parser ernie_x1 \
    --tool-parser-plugin FastDeploy/fastdeploy/entrypoints/openai/tool_parsers/ \
    --gpu-memory-utilization 0.9 \
    --load-choices "default_v1"
```

✅ FastDeploy **不需要自定义聊天模板** 来使用工具
其 tokenizer 和模板已支持 Qwen、ERNIE 等模型的工具调用格式。

### 客户端示例：触发工具调用

发送包含工具定义的请求，让模型根据需要选择调用工具：

```bash
#!/usr/bin/env python3
...
```

示例响应中的工具调用部分：

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

## 并行工具调用

如果模型能生成并行工具调用，FastDeploy 会返回一个列表：

```bash
tool_calls=[
  {"id": "...", "function": {...}},
  {"id": "...", "function": {...}}
]
```

FastDeploy 会按顺序完整透传所有调用，完全兼容 OpenAI 行为。

## 编写自定义工具解析器

FastDeploy 当前使用统一的 OpenAI 工具调用解析器。如果你想自定义插件，FastDeploy 提供了入口：

`fastdeploy/entrypoints/openai/tool_parser/`

自定义解析器需要实现：

```bash
class ToolParser:
    def extract(self, output_text: str) -> ToolCallResult:
        ...
```

注册方式：

```bash
ToolParserManager.register("my_parser", MyParser)
```

启用方式：

```bash
--tool-parser-plugin <插件文件的绝对路径>
--tool-call-parser my_parser
```

---

如需我帮你整理成双语对照版、排成更正式的文档、或合并到你已有的 FastDeploy 文档里，也可以继续告诉我。
