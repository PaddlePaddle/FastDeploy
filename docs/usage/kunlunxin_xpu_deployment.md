## Supported Models
|Model Name|Context Length|Quantization|XPUs Required|Deployment Commands|Minimum Version Required|
|-|-|-|-|-|-|
|ERNIE-4.5-300B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-300B-A47B|32K|WINT4|4 (Recommended)|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.0|
|ERNIE-4.5-300B-A47B|32K|WINT4|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.95 \ <br>    --load-choices "default"|>=2.0.0|
|ERNIE-4.5-300B-A47B|128K|WINT4|8 (Recommended)|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.0|
|ERNIE-4.5-21B-A3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|32K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|128K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-0.3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-0.3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-0.3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-0.3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # Specify any card<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|

## Quick start

### Online serving (OpenAI API-Compatible server)

Deploy an OpenAI API-compatible server using FastDeploy with the following commands:

#### Start service

**Deploy the ERNIE-4.5-300B-A47B-Paddle model with WINT4 precision and 32K context length on 4 XPUs**

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # Specify which cards to be used
export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1 is not supported
python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-300B-A47B-Paddle \
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
