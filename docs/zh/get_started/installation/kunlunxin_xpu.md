[English](../../../get_started/installation/kunlunxin_xpu.md)

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

```bash
mkdir Work
cd Work
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.3.0
docker run --name fastdeploy-xpu --net=host -itd --privileged -v $PWD:/Work -w /Work \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.3.0 \
    /bin/bash
docker exec -it fastdeploy-xpu /bin/bash
```

## 2. 使用 Pip 安装

### 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

或者您也可以安装最新版 PaddlePaddle（不推荐）

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### 安装 FastDeploy（**注意不要通过 pypi 源安装**）

```bash
python -m pip install fastdeploy-xpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

或者你也可以安装最新版 FastDeploy（不推荐）

```bash
python -m pip install --pre fastdeploy-xpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. 从源码编译安装

### 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
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
bash custom_ops/xpu_ops/download_dependencies.sh stable
```

或者你也可以下载最新版编译依赖

```bash
bash custom_ops/xpu_ops/download_dependencies.sh develop
```

设置环境变量

```bash
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
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

## 如何在昆仑芯 XPU 上部署服务
请参考 [**支持的模型与服务部署**](../../usage/kunlunxin_xpu_deployment.md) 以了解昆仑芯 XPU 支持的模型与服务部署方法。
