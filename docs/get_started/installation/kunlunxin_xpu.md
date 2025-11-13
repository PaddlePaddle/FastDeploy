[简体中文](../../zh/get_started/installation/kunlunxin_xpu.md)

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

```bash
mkdir Work
cd Work
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.3.0
docker run --name fastdeploy-xpu --net=host -itd --privileged -v $PWD:/Work -w /Work \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.3.0 \
    /bin/bash
docker exec -it fastdeploy-xpu /bin/bash
```

## 2. Set up using pre-built wheels

### Install PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

Alternatively, you can install the latest version of PaddlePaddle (Not recommended)

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### Install FastDeploy (**Do NOT install via PyPI source**)

```bash
python -m pip install fastdeploy-xpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Alternatively, you can install the latest version of FastDeploy (Not recommended)

```bash
python -m pip install --pre fastdeploy-xpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-xpu-p800/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. Build wheel from source

### Install PaddlePaddle

```bash
python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
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
bash custom_ops/xpu_ops/download_dependencies.sh stable
```

Alternatively, you can download the latest versions of XTDK and XVLLM (Not recommended)

```bash
bash custom_ops/xpu_ops/download_dependencies.sh develop
```

Set environment variables,

```bash
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
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

## How to deploy services on Kunlunxin XPU
Refer to [**Supported Models and Service Deployment**](../../usage/kunlunxin_xpu_deployment.md) for the details about the supported models and the way to deploy services on Kunlunxin XPU.
