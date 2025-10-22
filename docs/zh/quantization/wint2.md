[English](../../quantization/wint2.md)

# WINT2量化

权重经过 [CCQ（卷积编码量化）](https://arxiv.org/pdf/2507.07145) 方法进行离线压缩。权重的实际存储数值类型为INT8，每个INT8数值中打包了4个权重，等价于每个权重2bits。激活不做量化。在推理过程中，权重会被实时反量化并解码为BF16数值类型，并使用BF16数值类型进行计算。
- **支持硬件**：GPU
- **支持结构**：MoE结构

该方法依托卷积算法利用重叠的Bit位将2Bit的数值映射到更大的数值表示空间，使得模型权重量化后既保留原始数据更多的信息，同时将真实数值压缩到极低的2Bit大小，大致原理可参考下图：
![卷积编码量化示意图](./images/wint2.png)

CCQ WINT2一般用于资源受限的低门槛场景，以ERNIE-4.5-300B-A47B为例，将权重压缩到89GB，可支持141GB H20单卡部署。

## 执行WINT2离线推理
- 执行TP2/TP4模型时，可更换`model_name_or_path`以及`tensor_parallel_size`参数。
```
model_name_or_path = "baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle"
prompts = ["解析三首李白的诗"]
from fastdeploy import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0.7, top_p=0, max_tokens=128)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, use_cudagraph=True,)
outputs = llm.generate(prompts, sampling_params)
print(outputs)

```

## 启动WINT2推理服务
- 执行TP2/TP4模型时，可更换`--model`以及`tensor-parallel-size`参数；
```
python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --cache-queue-port 8183 \
    --tensor-parallel-size 1 \
    --max-model-len  32768 \
    --use-cudagraph \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256
```

## 用户发起服务请求

执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```
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

FastDeploy服务接口兼容OpenAI协议，可以通过如下Python代码发起服务请求。

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

通过指定 `--model baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle` 可自动从AIStudio下载已离线量化好的WINT2模型，在该模型的config.json文件中，会有WINT2量化相关的配置信息，不用再在启动推理服务时设置 `--quantization`.

模型的config.json文件中的量化配置示例如下：

```
"quantization_config": {
    "dense_quant_type": "wint8",
    "moe_quant_type": "w4w2",
    "quantization": "wint2",
    "moe_quant_config": {
    "moe_w4_quant_config": {
        "quant_type": "wint4",
        "quant_granularity": "per_channel",
        "quant_start_layer": 0,
        "quant_end_layer": 6
    },
    "moe_w2_quant_config": {
        "quant_type": "wint2",
        "quant_granularity": "pp_acc",
        "quant_group_size": 64,
        "quant_start_layer": 7,
        "quant_end_layer": 53
    }
  }
}
```

- 更多部署教程请参考[get_started](../get_started/ernie-4.5.md)；
- 更多模型说明请参考[支持模型列表](../supported_models.md)。

## WINT2效果

在ERNIE-4.5-300B-A47B模型上，WINT2与WINT4效果对比：

| 测试集 |数据集大小| WINT4 | WINT2 |
|---------|---------|---------|---------|
| IFEval |500|88.17 | 85.95 |
|BBH|6511|94.43|90.06|
|DROP|9536|91.17|89.32|
|CMMLU|11477|89.92|86.55|
