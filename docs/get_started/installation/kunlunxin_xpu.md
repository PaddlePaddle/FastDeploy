# Kunlunxin XPU

## Requirements

- OS: Linux
- Python: 3.10
- XPU Model: P800
- XPU Driver Version: ≥ 5.0.21.26
- XPU Firmware Version: ≥ 1.48

Verified platform:
- CPU: INTEL(R) XEON(R) PLATINUM 8563C / Hygon C86-4G 7490 64-core Processor
- Memory: 2T
- Disk: 4T
- OS: CentOS release 7.6 (Final)
- Python: 3.10
- XPU Model: P800 (OAM Edition)
- XPU Driver Version: 5.0.21.26
- XPU Firmware Version: 1.48

**Note:** Currently, only INTEL or Hygon CPU-based P800 (OAM Edition) servers have been verified. Other CPU types and P800 (PCIe Edition) servers have not been tested yet.

## 1. Set up using Docker (Recommended)

**This Docker image is ready-to-use with all dependencies pre-installed. No additional compilation required!**

### Basic Installation

```bash
mkdir Work
cd Work
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0
docker run --name fastdeploy-xpu --net=host -itd --privileged -v $PWD:/Work -w /Work \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0 \
    /bin/bash
docker exec -it fastdeploy-xpu /bin/bash
```

### Deploy ERNIE Models Directly (Ready-to-use Example)

If you want to deploy ERNIE models immediately and access them via OpenAI API, you can run directly inside the container:

```bash
# Inside the container, deploy ERNIE-4.5-0.3B model directly
export XPU_VISIBLE_DEVICES="0"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

Then you can access the model via OpenAI-compatible API:

```bash
# Test from host machine or another terminal
curl -X POST "http://localhost:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello, please introduce ERNIE large language model"}
  ]
}'
```

## 2. Set up using pre-built wheels

### Install PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

Alternatively, you can install the latest version of PaddlePaddle (Not recommended)

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### Install FastDeploy (**Do NOT install via PyPI source**)

```bash
python -m pip install fastdeploy-xpu==2.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Alternatively, you can install the latest version of FastDeploy (Not recommended)

```bash
python -m pip install --pre fastdeploy-xpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. Build wheel from source

### Install PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

Alternatively, you can install the latest version of PaddlePaddle (Not recommended)

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### Download FastDeploy source code, checkout the stable branch/TAG

```bash
git clone https://github.com/PaddlePaddle/FastDeploy
git checkout <tag or branch>
cd FastDeploy
```

### Download Kunlunxin Compilation Dependency

```bash
bash custom_ops/xpu_ops/src/download_dependencies.sh stable
```

Alternatively, you can download the latest versions of XTDK and XVLLM (Not recommended)

```bash
bash custom_ops/xpu_ops/src/download_dependencies.sh develop
```

Set environment variables,

```bash
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/src/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/src/third_party/xvllm
```

### Compile and Install.

```bash
bash build.sh
```

The compiled outputs will be located in the ```FastDeploy/dist``` directory.

## Installation verification

```bash
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddle.jit.marker import unified"
python -c "from fastdeploy.model_executor.ops.xpu import block_attn"
```

If all the above steps execute successfully, FastDeploy is installed correctly.

## Frequently Asked Questions

### Q: Is there a ready-to-use Docker image that doesn't require compiling dependencies inside the container?

**A: Yes! The recommended Docker image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0` is ready-to-use.**

This image comes pre-installed with:
- PaddlePaddle XPU version
- FastDeploy XPU version  
- All necessary Kunlunxin XPU dependencies
- XRE runtime environment

You can deploy ERNIE models directly inside the container without any additional compilation steps:

```bash
# Pull and run container
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0
docker run --name fastdeploy-xpu --net=host -itd --privileged \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0

# Enter container and deploy model directly
docker exec -it fastdeploy-xpu bash
export XPU_VISIBLE_DEVICES="0"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

Then call via OpenAI API:

```python
import openai
client = openai.Client(base_url="http://localhost:8188/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[{"role": "user", "content": "Hello, please introduce ERNIE large language model"}],
)
print(response.choices[0].message.content)
```

### Q: When do I need to build from source?

A: You only need to build from source when:
- You need the latest development version features
- You need to customize FastDeploy source code
- The Docker image version doesn't meet your requirements

For most users, we recommend using the Docker image directly.

## How to deploy services on Kunlunxin XPU
Refer to [**Supported Models and Service Deployment**](../../usage/kunlunxin_xpu_deployment.md) for the details about the supported models and the way to deploy services on Kunlunxin XPU.
