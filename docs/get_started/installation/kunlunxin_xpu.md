# Kunlunxin XPU

## Requirements

- OS: Linux
- Python: 3.10
- XPU Model: P800
- XPU Driver Version: ≥ 5.0.21.10
- XPU Firmware Version: ≥ 1.31

Verified platform:
- CPU: INTEL(R) XEON(R) PLATINUM 8563C
- Memory: 2T
- Disk: 4T
- OS: CentOS release 7.6 (Final)
- Python: 3.10
- XPU Model: P800 (OAM Edition)
- XPU Driver Version: 5.0.21.10
- XPU Firmware Version: 1.31

**Note:** Currently, only INTEL or Hygon CPU-based P800 (OAM Edition) servers have been verified. Other CPU types and P800 (PCIe Edition) servers have not been tested yet.

## 1. Set up using Docker (Recommended)

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.0.0
```

## 2. Set up using pre-built wheels

### Install PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

Alternatively, you can install the latest version of PaddlePaddle (Not recommended)

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### Install FastDeploy (**Do NOT install via PyPI source**)

```bash
python -m pip install fastdeploy-xpu==2.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

Alternatively, you can install the latest version of FastDeploy (Not recommended)

```bash
python -m pip install --pre fastdeploy-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

## 3. Build wheel from source

### Install PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

Alternatively, you can install the latest version of PaddlePaddle (Not recommended)

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### Download Kunlunxin Toolkit (XTDK) and XVLLM library, then set their paths.

```bash
# XTDK
wget https://klx-sdk-release-public.su.bcebos.com/xtdk_15fusion/dev/3.2.40.1/xtdk-llvm15-ubuntu2004_x86_64.tar.gz
tar -xvf xtdk-llvm15-ubuntu2004_x86_64.tar.gz && mv xtdk-llvm15-ubuntu2004_x86_64 xtdk
export CLANG_PATH=$(pwd)/xtdk

# XVLLM
wget https://klx-sdk-release-public.su.bcebos.com/xinfer/daily/eb/20250624/output.tar.gz
tar -xvf output.tar.gz && mv output xvllm
export XVLLM_PATH=$(pwd)/xvllm
```

Alternatively, you can download the latest versions of XTDK and XVLLM (Not recommended)

```bash
XTDK: https://klx-sdk-release-public.su.bcebos.com/xtdk_15fusion/dev/latest/xtdk-llvm15-ubuntu2004_x86_64.tar.gz
XVLLM: https://klx-sdk-release-public.su.bcebos.com/xinfer/daily/eb/latest/output.tar.gz
```

### Download FastDeploy source code, checkout the stable branch/TAG, then compile and install.

```bash
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
bash build.sh
```

The compiled outputs will be located in the ```FastDeploy/dist``` directory.

## Installation verification

```python
import paddle
from paddle.jit.marker import unified
paddle.utils.run_check()
from fastdeploy.model_executor.ops.xpu import block_attn
```

If all the above steps execute successfully, FastDeploy is installed correctly.

## Quick start

Currently, P800 has only validated deployment of the following models:
- ERNIE-4.5-300B-A47B-Paddle 32K WINT4 (8-card)
- ERNIE-4.5-300B-A47B-Paddle 128K WINT4 (8-card)

### Offline inference

After installing FastDeploy, you can perform offline text generation with user-provided prompts using the following code,

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "Where is the capital of China?",
]

sampling_params = SamplingParams(top_p=0.95)

llm = LLM(model="baidu/ERNIE-4.5-300B-A47B-Paddle", tensor_parallel_size=8, max_model_len=8192, quantization='wint4')

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
```

Refer to [Parameters](../../parameters.md) for more configuration options.

### Online serving (OpenAI API-Compatible server)

Deploy an OpenAI API-compatible server using FastDeploy with the following commands:

#### Start service

**ERNIE-4.5-300B-A47B-Paddle 32K WINT4 (8-card) (Recommended)**

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

**ERNIE-4.5-300B-A47B-Paddle 128K WINT4 (8-card)**

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --gpu-memory-utilization 0.9
```

Refer to [Parameters](../../parameters.md) for more options.

#### Send requests

Send requests using either curl or Python

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
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Where is the capital of China?"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

For detailed OpenAI protocol specifications, see [OpenAI Chat Compeltion API](https://platform.openai.com/docs/api-reference/chat/create). Differences from the standard OpenAI protocol are documented in [OpenAI Protocol-Compatible API Server](../../online_serving/README.md).
