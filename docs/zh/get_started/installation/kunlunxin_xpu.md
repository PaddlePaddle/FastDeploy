# 昆仑芯 XPU

## 要求

- OS：Linux
- Python：3.10
- XPU 型号：P800
- XPU 驱动版本：≥ 5.0.21.10
- XPU 固件版本：≥ 1.31

已验证的平台：
- CPU：INTEL(R) XEON(R) PLATINUM 8563C / Hygon C86-4G 7490 64-core Processor
- 内存：2T
- 磁盘：4T
- OS：CentOS release 7.6 (Final)
- Python：3.10
- XPU 型号：P800（OAM 版）
- XPU 驱动版本：5.0.21.10
- XPU 固件版本：1.31

**注：** 目前只验证过 INTEL 或海光 CPU OAM 版 P800 服务器，暂未验证其它 CPU 和 PCIe 版 P800 服务器。

## 1. 使用 Docker 安装（推荐）

```bash
mkdir Work
cd Work
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.0.0
docker run --name fastdeploy-xpu --net=host -itd --privileged -v $PWD:/Work -w /Work \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.0.0 \
    /bin/bash
docker exec -it fastdeploy-xpu /bin/bash
```

## 2. 使用 Pip 安装

### 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

或者您也可以安装最新版 PaddlePaddle（不推荐）

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### 安装 FastDeploy（**注意不要通过 pypi 源安装**）

```bash
python -m pip install fastdeploy-xpu==2.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

或者你也可以安装最新版 FastDeploy（不推荐）

```bash
python -m pip install --pre fastdeploy-xpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. 从源码编译安装

### 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

或者您也可以安装最新版 PaddlePaddle（不推荐）

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### 下载 FastDelpoy 源码，切换到稳定分支或 TAG

```bash
git clone https://github.com/PaddlePaddle/FastDeploy
git checkout <tag or branch>
cd FastDeploy
```

### 下载昆仑编译依赖

```bash
bash custom_ops/xpu_ops/src/download_dependency.sh stable
```

或者你也可以下载最新版编译依赖

```bash
bash custom_ops/xpu_ops/src/download_dependency.sh develop
```

设置环境变量

```bash
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/src/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/src/third_party/xvllm
```

### 开始编译并安装：

```bash

bash build.sh
```

编译后的产物在 ```FastDeploy/dist``` 目录下。

## 验证是否安装成功

```python
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddle.jit.marker import unified"
python -c "from fastdeploy.model_executor.ops.xpu import block_attn"
```

如果上述步骤均执行成功，代表 FastDeploy 已安装成功。

## 快速开始

P800 支持 ```ERNIE-4.5-300B-A47B-Paddle``` 模型采用以下配置部署（注意：不同配置在效果、性能上可能存在差异）。
- 32K WINT4 8 卡（推荐）
- 128K WINT4 8 卡
- 32K WINT4 4 卡

### OpenAI 兼容服务器

您还可以通过如下命令，基于 FastDeploy 实现 OpenAI API 协议兼容的服务器部署。

#### 启动服务

**基于 WINT4 精度和 32K 上下文部署 ERNIE-4.5-300B-A47B-Paddle 模型到 8 卡 P800 服务器（推荐）**

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

**基于 WINT4 精度和 128K 上下文部署 ERNIE-4.5-300B-A47B-Paddle 模型到 8 卡 P800 服务器**

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

**基于 WINT4 精度和 32K 上下文部署 ERNIE-4.5-300B-A47B-Paddle 模型到 4 卡 P800 服务器**

```bash
export XPU_VISIBLE_DEVICES="0,1,2,3" # 设置使用的 XPU 卡
python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-300B-A47B-Paddle \
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

更多参数可以参考 [参数说明](../../parameters.md)。

全部支持的模型可以在下面的 *支持的模型* 章节找到。

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

OpenAI 协议的更多说明可参考文档 [OpenAI Chat Compeltion API](https://platform.openai.com/docs/api-reference/chat/create)，以及与 OpenAI 协议的区别可以参考 [兼容 OpenAI 协议的服务化部署](../../online_serving/README.md)。

## 支持的模型
|模型名|上下文长度|量化|所需卡数|部署命令|
|-|-|-|-|-|
|ERNIE-4.5-300B-A47B|32K|WINT8|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-300B-A47B|32K|WINT4|4 （推荐）|export XPU_VISIBLE_DEVICES="0,1,2,3" or "4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 4 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-300B-A47B|32K|WINT4|8|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-300B-A47B|128K|WINT4|8 （推荐）|export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 8 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 64 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-21B-A3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-21B-A3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-21B-A3B|32K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-21B-A3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-21B-A3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-21B-A3B|128K|WINT4|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint4" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-0.3B|32K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-0.3B|32K|WINT8|1|export XPU_VISIBLE_DEVICES="x" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 32768 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-0.3B|128K|BF16|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --gpu-memory-utilization 0.9|
|ERNIE-4.5-0.3B|128K|WINT8|1|export XPU_VISIBLE_DEVICES="0" # 指定任意一张卡<br>python -m fastdeploy.entrypoints.openai.api_server \ <br>    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \ <br>    --port 8188 \ <br>    --tensor-parallel-size 1 \ <br>    --max-model-len 131072 \ <br>    --max-num-seqs 128 \ <br>    --quantization "wint8" \ <br>    --gpu-memory-utilization 0.9|
