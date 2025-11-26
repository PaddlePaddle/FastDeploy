[简体中文](../zh/get_started/ernie-4.5-vl-thinking.md)

# Deploy ERNIE-4.5-VL-28B-A3B-Thinking Multimodal Thinking Model

This document explains how to deploy the ERNIE-4.5-VL multimodal model, supporting user interaction via multimodal data and tool call (including for multimodal data). Ensure your hardware meets the requirements before deployment.

- GPU Driver >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 1 GPUs

Refer to the [Installation Guide](./installation/README.md) for FastDeploy setup.

## Prepare the Model
Specify ```--model baidu/ERNIE-4.5-VL-28B-A3B-Thinking``` during deployment to automatically download the model from AIStudio with resumable downloads. You can also manually download the model from other sources. Note that FastDeploy requires Paddle-format models. For more details, see [Supported Models](../supported_models.md).

## Launch the Service

Execute the following command to start the service. For parameter configurations, refer to [Parameter Guide](../parameters.md).

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

## Request the Service
After launching, the service is ready when the following logs appear:

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

### Health Check

Verify service status (HTTP 200 indicates success):

```shell
curl -i http://0.0.0.0:8180/health
```

### cURL Request
Send requests as follows:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Rewrite Li Bai's 'Quiet Night Thoughts' as a modern poem"}
  ]
}'
```
\
For image inputs:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"From which era does the artifact in the image originate?"}
    ]}
  ]
}'
```
Image can also be provided through base64-encoded string:
```shell
{"type":"image_url", "image_url": {"url":"data:image/jpg;base64,this/is/an/example"}
```
or absolute path to local file:
```shell
{"type":"image_url", "image_url": {"url":"file:///this/is/an/example"}
```
\
For video inputs:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"video_url", "video_url": {"url":"https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"}},
      {"type":"text", "text":"How many apples are in the scene?"}
    ]}
  ]
}'
```
Video can also be provided through base64-encoded string:
```shell
{"type":"video_url", "video_url": {"url":"data:video/mp4;base64,this/is/an/example"}
```
or absolute path to local file:
```shell
{"type":"video_url", "video_url": {"url":"file:///this/is/an/example"}
```
\
Input includes tool calls, send requests with the command below:

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

For multi-round requests with tool results in history context, use the command below:
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

### Python Client (OpenAI-compatible API)

FastDeploy's API is OpenAI-compatible. You can also use Python for streaming requests:

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
            {"type": "text", "text": "From which era does the artifact in the image originate?"},
        ]},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## Model Output
Example output with reasoning (reasoning content in `reasoning_content`, response in `content`, tool_calls in `tool_calls`):

Example of non-streaming results without tool call:
```json
{
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The artifact in the image ...",
                "multimodal_content": null,
                "reasoning_content": "The user asks about ...",
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

Example of non-streaming results with tool call, where the `content` field is empty and `finish_reason` is `tool_calls`:

```json
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
