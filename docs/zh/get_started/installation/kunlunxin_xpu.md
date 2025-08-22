# 昆仑芯 XPU

## 要求

- OS：Linux
- Python：3.10
- XPU 型号：P800
- XPU 驱动版本：≥ 5.0.21.26
- XPU 固件版本：≥ 1.48

已验证的平台：
- CPU：INTEL(R) XEON(R) PLATINUM 8563C / Hygon C86-4G 7490 64-core Processor
- 内存：2T
- 磁盘：4T
- OS：CentOS release 7.6 (Final)
- Python：3.10
- XPU 型号：P800（OAM 版）
- XPU 驱动版本：5.0.21.26
- XPU 固件版本：1.48

**注：** 目前只验证过 INTEL 或海光 CPU OAM 版 P800 服务器，暂未验证其它 CPU 和 PCIe 版 P800 服务器。

## 1. 使用 Docker 安装（推荐）

**此 Docker 镜像是开箱即用的，已预装所有必需依赖，无需额外编译！**

### 基础安装

```bash
mkdir Work
cd Work
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0
docker run --name fastdeploy-xpu --net=host -itd --privileged -v $PWD:/Work -w /Work \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0 \
    /bin/bash
docker exec -it fastdeploy-xpu /bin/bash
```

### 直接部署 ERNIE 模型（开箱即用示例）

如果您想直接部署 ERNIE 文心模型并通过 OpenAI API 接口访问，可以直接在容器内运行：

```bash
# 进入容器后，直接部署 ERNIE-4.5-0.3B 模型
export XPU_VISIBLE_DEVICES="0"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

然后您可以通过 OpenAI 兼容的 API 接口访问模型：

```bash
# 在宿主机或另一个终端中测试
curl -X POST "http://localhost:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "你好，请介绍一下文心大模型"}
  ]
}'
```

## 2. 使用 Pip 安装

### 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

或者您也可以安装最新版 PaddlePaddle（不推荐）

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### 安装 FastDeploy（**注意不要通过 pypi 源安装**）

```bash
python -m pip install fastdeploy-xpu==2.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

或者你也可以安装最新版 FastDeploy（不推荐）

```bash
python -m pip install --pre fastdeploy-xpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. 从源码编译安装

### 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
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
bash custom_ops/xpu_ops/src/download_dependencies.sh stable
```

或者你也可以下载最新版编译依赖

```bash
bash custom_ops/xpu_ops/src/download_dependencies.sh develop
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

## 常见问题解答

### Q: 有没有开箱即用的镜像，不需要在容器中编译依赖？

**A: 有的！推荐的 Docker 镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0` 就是开箱即用的。**

这个镜像已经预装了：
- PaddlePaddle XPU 版本
- FastDeploy XPU 版本  
- 所有必需的昆仑芯 XPU 依赖
- XRE 运行时环境

您可以直接在容器内部署 ERNIE 文心模型，无需任何额外的编译步骤：

```bash
# 拉取并运行容器
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0
docker run --name fastdeploy-xpu --net=host -itd --privileged \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.1.0

# 进入容器直接部署模型
docker exec -it fastdeploy-xpu bash
export XPU_VISIBLE_DEVICES="0"
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/ERNIE-4.5-0.3B-Paddle \
    --port 8188 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

然后通过 OpenAI API 接口调用：

```python
import openai
client = openai.Client(base_url="http://localhost:8188/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[{"role": "user", "content": "你好，请介绍一下文心大模型"}],
)
print(response.choices[0].message.content)
```

### Q: 什么时候需要从源码编译？

A: 只有在以下情况下才需要从源码编译：
- 需要最新的开发版本功能
- 需要自定义修改 FastDeploy 源码
- Docker 镜像版本不满足需求

对于大多数用户，推荐直接使用 Docker 镜像。

## 如何在昆仑芯 XPU 上部署服务
请参考 [**支持的模型与服务部署**](../../usage/kunlunxin_xpu_deployment.md) 以了解昆仑芯 XPU 支持的模型与服务部署方法。
