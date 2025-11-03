[简体中文](../zh/usage/kunlunxin_xpu_deployment.md)

## Supported Models
|Model Name|Context Length|Quantization|XPUs Required|Deployment Commands|Applicable Version|
|-|-|-|-|-|-|
|ERNIE-4.5-300B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-300B-A47B|32K|WINT4|4 (Recommended)|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-300B-A47B|32K|WINT4|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.95 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-300B-A47B|128K|WINT4|8 (Recommended)|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-21B-A3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-21B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-21B-A3B|32K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-21B-A3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-21B-A3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-21B-A3B|128K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-0.3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-0.3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-0.3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-0.3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle|64K|W4A8|4|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "W4A8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|2.3.0|
|ERNIE-4.5-VL-28B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle \ <br>    --port 8188  \ <br> --tensor-parallel-size 1 \ <br> --quantization "wint8" \ <br>  --max-model-len 32768 \ <br> --max-num-seqs 10 \ <br>     --enable-mm \ <br>   --mm-processor-kwargs '{"video_max_frames": 30}' \ <br>     --limit-mm-per-prompt '{"image": 10, "video": 3}' \ <br>     --reasoning-parser ernie-45-vl |2.3.0|
|ERNIE-4.5-VL-424B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" <br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Paddle \ <br>    --port 8188 \ <br> --tensor-parallel-size 8 \ <br> --quantization "wint8" \ <br>  --max-model-len 32768 \ <br> --max-num-seqs 10 \ <br>     --enable-mm \ <br>   --mm-processor-kwargs '{"video_max_frames": 30}' \ <br>     --limit-mm-per-prompt '{"image": 10, "video": 3}' \ <br>     --reasoning-parser ernie-45-vl |2.3.0|

## Quick start

### Deploy online serving based on ERNIE-4.5-300B-A47B-Paddle

#### Start service

Deploy the ERNIE-4.5-300B-A47B-Paddle model with WINT4 precision and 32K context length on 4 XPUs

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # Specify which cards to be used
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9 \
    --load-choices "default"
```

**Note:** When deploying on 4 XPUs, only two configurations are supported which constrained by hardware limitations such as interconnect capabilities.
`export XPU_VISIBLE_DEVICES="0,1,2,3"`
or
`export XPU_VISIBLE_DEVICES="4,5,6,7"`

Refer to [Parameters](../parameters.md) for more options.

All supported models can be found in the *Supported Models* section above.

#### Send requests
Send requests using either curl or Python.

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

For detailed OpenAI protocol specifications, see [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create). Differences from the standard OpenAI protocol are documented in [OpenAI Protocol-Compatible API Server](../online_serving/README.md).

### Deploy online serving based on ERNIE-4.5-VL-28B-A3B-Paddle

#### Start service
Deploy the ERNIE-4.5-VL-28B-A3B-Paddle model with WINT8 precision and 32K context length on 1 XPU

```bash
export XPU_VISIBLE_DEVICES="0" # Specify any card
python -m fastdeploy.entrypoints.openai.api_server \
--model PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle \
--port 8188  \
--tensor-parallel-size 1 \
--quantization "wint8" \
--max-model-len 32768 \
--max-num-seqs 10 \
--enable-mm \
--mm-processor-kwargs '{"video_max_frames": 30}' \
--limit-mm-per-prompt '{"image": 10, "video": 3}' \
--reasoning-parser ernie-45-vl
```

#### Send requests

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
        is_reason = "[answer]" if reasoning_content == '' else "[think]"
        is_reason = ""
        print(reasoning_content+content+is_reason, end='', flush=True)
print('\n')
```
