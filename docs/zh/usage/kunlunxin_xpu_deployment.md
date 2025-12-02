[English](../../usage/kunlunxin_xpu_deployment.md)

## 支持的模型
|模型名|上下文长度|量化|所需卡数|部署命令|适用版本|
|-|-|-|-|-|-|
|ERNIE-4.5-300B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-300B-A47B|32K|WINT4|4 （推荐）|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-300B-A47B|32K|WINT4|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.95|2.3.0|
|ERNIE-4.5-300B-A47B|128K|WINT4|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-21B-A3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-21B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-21B-A3B|32K|WINT4|1 （推荐）|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-21B-A3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-21B-A3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-21B-A3B|128K|WINT4|1 （推荐）|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-0.3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-0.3B|32K|WINT8|1 （推荐）|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-0.3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-0.3B|128K|WINT8|1 （推荐）|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-300B-A47B-W4A8C8-TP4|32K|W4A8|4|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "W4A8" \ <br>    --gpu-memory-utilization 0.9|2.3.0|
|ERNIE-4.5-VL-28B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0"# 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle \ <br>    --port 8188  \ <br> --tensor-parallel-size 1 \ <br> --quantization "wint8" \ <br>  --max-model-len 32768 \ <br> --max-num-seqs 10 \ <br>     --enable-mm \ <br>   --mm-processor-kwargs '{"video_max_frames": 30}' \ <br>     --limit-mm-per-prompt '{"image": 10, "video": 3}' \ <br>     --reasoning-parser ernie-45-vl|2.3.0|
|ERNIE-4.5-VL-424B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" <br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Paddle \ <br>    --port 8188 \ <br> --tensor-parallel-size 8 \ <br> --quantization "wint8" \ <br>  --max-model-len 32768 \ <br> --max-num-seqs 8 \ <br>     --enable-mm \ <br>   --mm-processor-kwargs '{"video_max_frames": 30}' \ <br>     --limit-mm-per-prompt '{"image": 10, "video": 3}' \ <br>     --reasoning-parser ernie-45-vl \ <br>   --gpu-memory-utilization 0.7|2.3.0|
|PaddleOCR-VL-0.9B|32K|BF16|1|export FD_ENABLE_MAX_PREFILL=1 <br>export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡 <br>python -m fastdeploy.entrypoints.openai.api_server \ <br>   --model PaddlePaddle/PaddleOCR-VL \ <br>  --port 8188 \ <br> --metrics-port 8181 \ <br> --engine-worker-queue-port 8182 \ <br> --max-model-len 16384 \ <br> --max-num-batched-tokens 16384 \ <br> --gpu-memory-utilization 0.8 \ <br> --max-num-seqs 256|2.3.0|
|ERNIE-4.5-VL-28B-A3B-Thinking|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br> --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking \ <br> --port 8188 \ <br> --tensor-parallel-size 1 \ <br> --quantization "wint8" \ <br> --max-model-len 131072 \ <br> --max-num-seqs 32 \ <br> --engine-worker-queue-port 8189 \ <br> --metrics-port 8190 \ <br> --cache-queue-port 8191 \ <br> --reasoning-parser ernie-45-vl-thinking \ <br> --tool-call-parser ernie-45-vl-thinking \ <br> --mm-processor-kwargs '{"image_max_pixels": 12845056 }'|2.3.0|

## 快速开始

### 基于ERNIE-4.5-300B-A47B-Paddle模型部署在线服务

#### 启动服务

基于 WINT4 精度和 32K 上下文部署 ERNIE-4.5-300B-A47B-Paddle 模型到 4 卡 P800 服务器

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # 设置使用的 XPU 卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**注意：** 使用 P800 在 4 块 XPU 上进行部署时，由于受到卡间互联拓扑等硬件限制，仅支持以下两种配置方式：
`export XPU_VISIBLE_DEVICES="0,1,2,3"`
or
`export XPU_VISIBLE_DEVICES="4,5,6,7"`

更多参数可以参考 [参数说明](../parameters.md)。

全部支持的模型可以在上方的 *支持的模型* 章节找到。

#### 请求服务

您可以基于 OpenAI 协议，通过 curl 和 python 两种方式请求服务。

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Where is the capital of China?"}
  ]
}'
```

```python
import openai
host = "0.0.0.0"
port = "8188"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.completions.create(
    model="null",
    prompt="Where is the capital of China?",
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].text, end='')
print('\n')

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "Where is the capital of China?"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

OpenAI 协议的更多说明可参考文档 [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create)，以及与 OpenAI 协议的区别可以参考 [兼容 OpenAI 协议的服务化部署](../online_serving/README.md)。

### 基于ERNIE-4.5-VL-28B-A3B-Paddle模型部署在线服务

#### 启动服务

基于 WINT8 精度和 32K 上下文部署 ERNIE-4.5-VL-28B-A3B-Paddle 模型到 单卡 P800 服务器

```bash
export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --quantization "wint8" \
    --max-model-len 32768 \
    --max-num-seqs 10 \
    --enable-mm \
    --mm-processor-kwargs '{"video_max_frames": 30}' \
    --limit-mm-per-prompt '{"image": 10, "video": 3}' \
    --reasoning-parser ernie-45-vl
```

#### 请求服务

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg", "detail": "high"}},
              {"type": "text", "text": "请描述图片内容"}
            ]}
    ],
    "metadata": {"enable_thinking": false}
}'
```

```python
import openai

ip = "0.0.0.0"
service_http_port = "8188"
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg", "detail": "high"}},
              {"type": "text", "text": "请描述图片内容"}
            ]
        },
    ],
    temperature=0.0001,
    max_tokens=10000,
    stream=True,
    top_p=0,
    metadata={"enable_thinking": False},
)

def get_str(content_raw):
    content_str = str(content_raw) if content_raw is not None else ''
    return content_str

for chunk in response:
    if chunk.choices[0].delta is not None and chunk.choices[0].delta.role != 'assistant':
        reasoning_content = get_str(chunk.choices[0].delta.reasoning_content)
        content = get_str(chunk.choices[0].delta.content)
        print(reasoning_content + content, end='', flush=True)
print('\n')
```

### 基于PaddleOCR-VL-0.9B模型部署在线服务

#### 启动服务

基于 BF16 精度和 16K 上下文部署 PaddleOCR-VL-0.9B 模型到 单卡 P800 服务器

```bash
export FD_ENABLE_MAX_PREFILL=1
export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
   --model PaddlePaddle/PaddleOCR-VL \
   --port 8188 \
   --metrics-port 8181 \
   --engine-worker-queue-port 8182 \
   --max-model-len 16384 \
   --max-num-batched-tokens 16384 \
   --gpu-memory-utilization 0.8 \
   --max-num-seqs 256
```

#### 请求服务

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo.jpg"}},
              {"type": "text", "text": "OCR:"}
            ]}
    ],
    "metadata": {"enable_thinking": false}
}'
```

```python
import openai

ip = "0.0.0.0"
service_http_port = "8188"
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo.jpg"}},
              {"type": "text", "text": "OCR:"}
            ]
        },
    ],
    temperature=0.0001,
    max_tokens=4096,
    stream=True,
    top_p=0,
    metadata={"enable_thinking": False},
)

def get_str(content_raw):
    content_str = str(content_raw) if content_raw is not None else ''
    return content_str

for chunk in response:
    if chunk.choices[0].delta is not None and chunk.choices[0].delta.role != 'assistant':
        reasoning_content = get_str(chunk.choices[0].delta.reasoning_content)
        content = get_str(chunk.choices[0].delta.content)
        print(reasoning_content + content, end='', flush=True)
print('\n')
```

### 基于ERNIE-4.5-VL-28B-A3B-Thinking模型部署在线服务

#### 启动服务

基于 WINT8 精度和 128K 上下文部署 ERNIE-4.5-VL-28B-A3B-Thinking 模型到 单卡 P800 服务器

```bash
export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking \
    --port 8188 \
    --tensor-parallel-size 1 \
    --quantization "wint8" \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --engine-worker-queue-port 8189 \
    --metrics-port 8190 \
    --cache-queue-port 8191 \
    --reasoning-parser ernie-45-vl-thinking \
    --tool-call-parser ernie-45-vl-thinking \
    --mm-processor-kwargs '{"image_max_pixels": 12845056 }'
```

### 请求服务
通过如下命令发起服务请求
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```
输入包含图片时，按如下命令发起请求
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
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
输入包含视频时，按如下命令发起请求
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
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
输入包含工具调用时，按如下命令发起请求
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
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
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
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
