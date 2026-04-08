[English](../../get_started/ernie-4.5-vl-thinking.md)

# ERNIE-4.5-VL-28B-A3B-Thinking 多模态思考模型

本文档讲解如何部署ERNIE-4.5-VL-28B-A3B-Thinking多模态思考模型，支持用户使用多模态数据与模型进行对话交互，同时支持工具调用能力（包括多模态数据）。在开始部署前，请确保你的硬件环境满足如下条件：

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 1卡

安装FastDeploy方式参考[安装文档](./installation/README.md)。

## 准备模型
部署时指定```--model baidu/ERNIE-4.5-VL-28B-A3B-Thinking```即可自动从AIStudio下载模型，并支持断点续传。你也可以自行从不同渠道下载模型，需要注意的是FastDeploy依赖Paddle格式的模型，更多说明参考[支持模型列表](../supported_models.md)。

## 启动服务

执行如下命令，启动服务,其中启动命令配置方式参考[参数说明](../parameters.md)

```shell
python -m fastdeploy.entrypoints.openai.api_server \
        --model baidu/ERNIE-4.5-VL-28B-A3B-Thinking \
        --max-model-len 131072 \
        --max-num-seqs 32 \
        --port 8180 \
        --quantization wint8 \
        --reasoning-parser ernie-45-vl-thinking \
        --tool-call-parser ernie-45-vl-thinking \
        --mm-processor-kwargs '{"image_max_pixels": 12845056 }'
```

## 用户发起服务请求
执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

FastDeploy提供服务探活接口，用以判断服务的启动状态，执行如下命令返回 ```HTTP/1.1 200 OK``` 即表示服务启动成功。

```shell
curl -i http://0.0.0.0:8180/health
```

通过如下命令发起服务请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```
\
输入包含图片时，按如下命令发起请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"图中的文物属于哪个年代?"}
    ]}
  ]
}'
```
图片url字段同样支持传入base64编码字符串:
```shell
{"type":"image_url", "image_url": {"url":"data:image/jpg;base64,this/is/an/example"}
```
或本地文件的绝对路径:
```shell
{"type":"image_url", "image_url": {"url":"file:///this/is/an/example"}
```
\
输入包含视频时，按如下命令发起请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"video_url", "video_url": {"url":"https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"}},
      {"type":"text", "text":"画面中有几个苹果?"}
    ]}
  ]
}'
```
视频url字段同样支持传入base64编码字符串:
```shell
{"type":"video_url", "video_url": {"url":"data:video/mp4;base64,this/is/an/example"}
```
或本地文件的绝对路径:
```shell
{"type":"video_url", "video_url": {"url":"file:///this/is/an/example"}
```
\
输入包含工具调用时，按如下命令发起请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d $'{
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            },
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner, and the values of x1, y1, x2, y2 are all normalized to the range 0–1000 based on the original image dimensions."
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional)."
                        }
                    },
                    "required": [
                        "bbox_2d"
                    ]
                },
                "strict": false
            }
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is the old lady on the left side of the empty table behind older couple?"
                }
            ]
        }
    ],
    "stream": false
}'
```

多轮请求， 历史上下文中包含工具返回结果时，按如下命令发起请求
```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d $'{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Get the current weather in Beijing"
                }
            ]
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
            "content": ""
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": "location: Beijing，temperature: 23，weather: sunny，unit: c"
                }
            ]
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

FastDeploy服务接口兼容OpenAI协议，可以通过如下Python代码发起服务请求, 以下示例开启流式用法。

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
            {"type": "text", "text": "图中的文物属于哪个年代?"},
        ]},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## 模型输出
模型生成内容中， 思考内容在 `reasoning_content` 字段中, 模型回复内容在 `content` 字段中， 工具调用在 `tool_calls` 字段中。

非流式无工具调用结果示例：
```json
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\n\n图中的文物是**北朝**时期的佛教造像（约公元420 - 589年）。  \n\n从造像风格来看，其背光形制、佛像面相特征（如慈祥的神情、面部轮廓）、服饰样式（通肩式袈裟）、胁侍菩萨的配置，以及整体雕刻技法（如莲座的莲瓣处理、背光区域的装饰纹样）等，都符合北朝（涵盖北魏、东魏、西魏、北齐、北周等政权）佛教造像的典型艺术特征。北朝是佛教艺术在中国北方蓬勃发展的时期，这类石造像在形制与审美上融合了西域佛教艺术传统与中原文化审美，是研究该阶段宗教、艺术与社会文化的重要实物资料。",
                "multimodal_content": null,
                "reasoning_content": "用户现在需要判断这尊佛像的年代。首先看风格：背光、造像特征。这应该是北朝（北魏、北周等）的佛教造像，尤其是北朝时期的石造像，风格上背光有繁复的装饰，佛像的面相、服饰（通肩式袈裟）、胁侍菩萨等元素。北朝佛教造像盛行，尤其是北魏迁都后，造像风格从西域向中原过渡，这尊像的背光造型、雕刻技法（如莲座的样式、胁侍的配置）符合北朝（约420 - 589年）的特征。需要确认典型特征：背光是舟形或火穗形？这里背光是类似舟形，装饰莲瓣、飞天等，佛像结跏趺坐，双手施无畏印，胁侍菩萨，这些都是北朝石造像的常见元素。所以判断为北朝时期（公元420 - 589年）。\n",
                "tool_calls": null
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1290,
        "total_tokens": 1681,
        "completion_tokens": 391,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "image_tokens": 1240,
            "video_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 217,
            "image_tokens": 0
        }
    }
}
```

非流式有工具结果调用示例， 其中 `content` 字段为空, 且 `finish_reason` 为 `tool_calls`：

```python
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "multimodal_content": null,
                "reasoning_content": "What immediately stands out is that I need to determine the spatial relationship between the old lady, the empty table, and the older couple. The original image might not provide enough detail to make this determination clearly, so I should use the image_zoom_in_tool to focus on the relevant area where these elements are located.\n",
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-dd0ef62027cf409c8f013af65f88adc3",
                        "type": "function",
                        "function": {
                            "name": "image_zoom_in_tool",
                            "arguments": "{\"bbox_2d\": [285, 235, 999, 652]}"
                        }
                    }
                ]
            }
            "finish_reason": "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 280,
        "total_tokens": 397,
        "completion_tokens": 117,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "image_tokens": 0,
            "video_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 66,
            "image_tokens": 0
        }
    }
}
```
