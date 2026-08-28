[English](../../get_started/minicpm41.md)

# MiniCPM4.1-8B模型

本文档讲解如何使用FastDeploy部署MiniCPM4.1-8B BF16模型或启用在线 WINT4/WINT8 量化。在开始部署前，请确保硬件环境满足如下条件：

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 48GB NVIDIA GPU 1卡

安装FastDeploy方式参考[安装文档](./installation/README.md)。

## 最小验收准备

进入 FastDeploy 仓库，只需指定本地模型目录和测试 GPU；脚本会自动使用项目 `.venv` 并配置其 CUDA 运行库：

```shell
cd /path/to/FastDeploy
export MODEL_PATH=/path/to/MiniCPM4.1-8B
export CUDA_VISIBLE_DEVICES=0
```

## 准备模型

### 1. 手动下载 （可选）

MiniCPM4.1-8B使用Hugging Face Torch safetensors格式。执行如下命令下载模型：

```shell
export MODEL_PATH=/path/to/MiniCPM4.1-8B
hf download openbmb/MiniCPM4.1-8B --local-dir "${MODEL_PATH}"
```

模型配置中的 `pad_token_id` 为2。部署前需要确保 `tokenizer_config.json` 中的 `pad_token` 为token ID 2对应的 `</s>`，可执行如下命令完成配置：

```shell
MODEL_PATH="${MODEL_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["MODEL_PATH"]) / "tokenizer_config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["pad_token"] = "</s>"
config_path.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

### 2. 自动下载

将模型名称设置为 `openbmb/MiniCPM4.1-8B`：

```shell
export MODEL_PATH=openbmb/MiniCPM4.1-8B
```

>💡 **注意**：如果当前目录下不存在 `--model` 指定的路径，FastDeploy会根据模型名称查询AIStudio预置模型；查询成功后自动下载，并支持断点续传。默认下载源为AIStudio，可通过 `FD_MODEL_SOURCE` 和 `FD_MODEL_CACHE` 配置下载源及缓存目录，详情参阅[模型下载](../../supported_models.md)。如果已经下载模型，也可以将 `MODEL_PATH` 设置为本地模型目录。

## 编译与最小回归

### 编译 CUDA 算子

完成 CUDA 算子编译和 InfLLM-V2 符号检查：

```shell
bash tests/benchmarks/test_minicpm41.sh build
```

脚本内部调用 `build.sh`。其他 GPU 可通过 `FD_BUILDING_ARCS` 调整架构。验收成功时输出 `PASS build`，并确认产物 `fastdeploy/model_executor/ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so` 包含以下符号：

- `infllmv2_update_compressed_k`
- `infllmv2_select_blocks`
- `infllmv2_attention_forward`

### MiniCPM4.1、thinking 与量化测试

测试模型注册、权重映射、多 token thinking、混合思考模式、WINT4/WINT8 在线量化和 InfLLM-V2 后端：

```shell
bash tests/benchmarks/test_minicpm41.sh unit
```

运行编译后的 CUDA 算子正确性测试：

```shell
bash tests/benchmarks/test_minicpm41.sh operators
```

通过标准为命令退出码0。

### E2E 服务测试

一条命令依次验收 BF16、WINT4、WINT8 和 InfLLM-V2：

```shell
bash tests/benchmarks/test_minicpm41.sh e2e
```

E2E 会自动选择临时端口、启动和清理服务，并在每种模式运行5个 `test_minicpm41_serving.py` 用例。每种模式显示 `5 passed` 即通过。

如需从编译到 E2E 全部一次执行：

```shell
bash tests/benchmarks/test_minicpm41.sh
```

## 启动服务

>💡 **注意**：以下命令使用单卡BF16模型，并关闭prefix caching。

执行如下命令启动服务，其中启动命令配置方式参考[参数说明](../parameters.md)。

```shell
export CUDA_VISIBLE_DEVICES=0
export FD_ATTENTION_BACKEND=FLASH_ATTN

python -m fastdeploy.entrypoints.openai.api_server \
       --model "${MODEL_PATH}" \
       --served-model-name MiniCPM4.1-8B \
       --port 8180 --engine-worker-queue-port 8182 \
       --cache-queue-port 8183 --metrics-port 8181 \
       --tensor-parallel-size 1 \
       --max-model-len 8192 \
       --max-num-seqs 1 \
       --max-num-batched-tokens 128 \
       --no-enable-prefix-caching
```

### 在线 WINT4/WINT8

继续使用同一个原始 BF16 checkpoint，并增加 `--quantization wint4` 或 `--quantization wint8`。FastDeploy 会在 BF16 权重加载后于内存中逐个量化 Linear 权重；MiniCPM4.1 的这条路径不需要、也不支持预先转换好的量化 checkpoint。其余服务参数与上面的 BF16 命令相同。

完成前面的最小验收准备后，启动 WINT4：

```shell
bash scripts/run_minicpm41_wint_server.sh wint4 "${MODEL_PATH}"
```

将 `wint4` 替换为 `wint8` 即可启用在线 INT8 权重量化。该脚本要求 `MODEL_PATH` 指向本地模型目录，并自动使用项目 `.venv`、CUDA 运行库及单卡安全默认参数。额外的 Server 参数可继续追加在命令末尾。若端口冲突，应同时为 API、metrics、engine queue 和 cache queue 更换一组未占用端口。公共 WINT 行为见[在线量化](../quantization/online_quantization.md)。

## 用户发起服务请求

执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```shell
INFO api_server.py[line:1030] Launching metrics service at http://0.0.0.0:8181/metrics
INFO api_server.py[line:1033] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
INFO api_server.py[line:1034] Launching completion service at http://0.0.0.0:8180/v1/completions
[INFO] Starting gunicorn 26.0.0
[INFO] Listening at: http://0.0.0.0:8180
[INFO] Application startup complete.
```

FastDeploy提供服务探活接口，用以判断服务的启动状态，执行如下命令返回 `HTTP/1.1 200 OK` 即表示服务启动成功。

```shell
curl -i http://0.0.0.0:8180/health
```

通过如下命令进行服务请求。`enable_thinking=false` 表示关闭思考模式。

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "MiniCPM4.1-8B",
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 64,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}' |  jq --indent 4 .
```

response如下：

```json

{
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "好的，这是将李白的《静夜思》改写为现代诗的版本：\n\n**《夜思》**\n\n窗外，月光\n如流水般倾泻\n\n白茫茫\n铺满了冰冷的窗棂\n\n远方的家\n在记忆深处闪烁\n\n小小的床\n"
            },
            "logprobs": null,
            "draft_logprobs": null,
            "prompt_logprobs": null,
            "finish_reason": "length",
            "speculate_metrics": null
        }
    ],
    "usage": {
        "prompt_tokens": 29,
        "total_tokens": 93,
        "completion_tokens": 64,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "image_tokens": 0,
            "video_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "image_tokens": 0
        }
    }
}
```

FastDeploy服务接口兼容OpenAI协议，可以通过如下Python代码发起服务请求。

```python
import openai

host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="MiniCPM4.1-8B",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ],
    temperature=0,
    top_p=1,
    max_tokens=64,
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\n")
```

MiniCPM4.1支持思考与非思考两种模式。将请求中的 `chat_template_kwargs.enable_thinking` 设置为 `true` 即可开启思考模式，还可以通过 `reasoning_max_tokens` 限制本次请求使用的思考token数。
