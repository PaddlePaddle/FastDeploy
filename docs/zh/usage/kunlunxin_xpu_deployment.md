## 支持的模型
|模型名|上下文长度|量化|所需卡数|部署命令|最低版本要求|
|-|-|-|-|-|-|
|ERNIE-4.5-300B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-300B-A47B|32K|WINT4|4 （推荐）|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.0|
|ERNIE-4.5-300B-A47B|32K|WINT4|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.95 \ <br>    --load-choices "default"|>=2.0.0|
|ERNIE-4.5-300B-A47B|128K|WINT4|8 （推荐）|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.0|
|ERNIE-4.5-21B-A3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|32K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-21B-A3B|128K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.1.0|
|ERNIE-4.5-0.3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-0.3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="x" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-0.3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|
|ERNIE-4.5-0.3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9 \ <br>    --load-choices "default"|>=2.0.3|

## 快速开始

### OpenAI 兼容服务器

您还可以通过如下命令，基于 FastDeploy 实现 OpenAI API 协议兼容的服务器部署。

#### 启动服务

**基于 WINT4 精度和 32K 上下文部署 ERNIE-4.5-300B-A47B-Paddle 模型到 4 卡 P800 服务器**

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # 设置使用的 XPU 卡
export ENABLE_V1_KVCACHE_SCHEDULER=0 # V1不支持
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

**注意：** 使用 P800 在 4 块 XPU 上进行部署时，由于受到卡间互联拓扑等硬件限制，仅支持以下两种配置方式：
`export XPU_VISIBLE_DEVICES="0,1,2,3"`
or
`export XPU_VISIBLE_DEVICES="4,5,6,7"`

更多参数可以参考 [参数说明](../../parameters.md)。

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
