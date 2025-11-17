[English](../../../get_started/installation/nvidia_gpu.md)

# NVIDIA CUDA GPU Installation

在环境满足如下条件前提下

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Python >= 3.10
- Linux X86_64

可通过如下5种方式进行安装

## 1. 预编译Docker安装(推荐)

**注意**： 如下镜像仅支持SM 80/90架构GPU（A800/H800等），如果你是在L20/L40/4090等SM 86/89架构的GPU上部署，请在创建容器后，卸载```fastdeploy-gpu```再重新安装如下文档指定支持86/89架构的`fastdeploy-gpu`包。

``` shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.3.0
```

## 2. 预编译Pip安装

首先安装 paddlepaddle-gpu，详细安装方式参考 [PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)

``` shell
# Install stable release
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install latest Nightly build
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
```

再安装 fastdeploy，**注意不要通过pypi源安装**，需要通过如下方式安装

如你的 GPU 是 SM80/90 架构(A100/H100等)，按如下方式安装

```
# 安装稳定版本fastdeploy
python -m pip install fastdeploy-gpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装Nightly Build的最新版本fastdeploy
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

如你的 GPU 是 SM86/89 架构(4090/L20/L40等)，按如下方式安装

```
# 安装稳定版本fastdeploy
python -m pip install fastdeploy-gpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装Nightly Build的最新版本fastdeploy
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. 镜像自行构建

> 注意 ```dockerfiles/Dockerfile.gpu``` 默认编译的架构支持SM 80/90，如若需要支持其它架构，需自行修改Dockerfile中的 ```bash build.sh 1 python false [80,90]```，建议不超过2个架构。

```
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

docker build -f dockerfiles/Dockerfile.gpu -t fastdeploy:gpu .
```

## 4. Wheel包源码编译

首先安装 paddlepaddle-gpu，详细安装方式参考 [PaddlePaddle安装](https://www.paddlepaddle.org.cn/)

``` shell
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

接着克隆源代码，编译安装

``` shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

# 第1个参数: 表示是否要构建wheel包，1表示打包，0表示只编译
# 第2个参数: Python解释器路径
# 第3个参数: 是否编译CPU推理算子
# 第4个参数: 编译的GPU架构
bash build.sh 1 python false [80,90]
```

编译后的产物在```FastDeploy/dist```目录下。

## 5. 算子预编译 Wheel 包

FastDeploy 提供了 GPU 算子预编译版 Wheel 包，可在无需完整源码编译的情况下快速构建。该方式当前仅支持 **SM90 架构（H20/H100等）** 和 **CUDA 12.6** 环境。

>默认情况下，`build.sh` 会从源码编译；若希望使用预编译包，可使用`FD_USE_PRECOMPILED` 参数；
>若预编译包下载失败或与环境不匹配，系统会自动回退至 `4. wheel 包源码编译` 模式。

首先安装 paddlepaddle-gpu，详细安装方式参考 [PaddlePaddle安装](https://www.paddlepaddle.org.cn/)

``` shell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

接着克隆源代码，拉取 whl 包并安装

```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

# 第1个参数: 是否打包成 wheel (1 表示打包)
# 第2个参数: Python 解释器路径
# 第3个参数: 是否编译 CPU 推理算子 (false 表示仅 GPU)
# 第4个参数: GPU 架构 (当前仅支持 [90])
# 第5个参数: 是否使用预编译算子 (1 表示启用预编译)
# 第6个参数(可选): 指定预编译算子的 commitID（默认使用当前的 commitID）

# 使用预编译 whl 包加速构建
bash build.sh 1 python false [90] 1

# 从指定 commitID 获取对应预编译算子
bash build.sh 1 python false [90] 1 8a9e7b53af4a98583cab65e4b44e3265a93e56d2
```

下载的 whl 包在 `FastDeploy/pre_wheel`目录下。

构建完成后，算子相关的产物位于 `FastDeploy/fastdeploy/model_executor/ops/gpu` 目录下。

> **说明：**
> - 该模式会优先下载预编译的 GPU 算子 whl 包，减少编译时间；
> - 目前仅支持 **GPU + SM90 + CUDA 12.6**；
> - 若希望自定义架构或修改算子逻辑，请使用 **源码编译方式（第4节）**。
> - 您可以在 FastDeploy CI 构建状态页面查看对应 commit 的预编译 whl 是否已构建成功。

## 环境检查

在安装 FastDeploy 后，通过如下 Python 代码检查环境的可用性

``` python
import paddle
from paddle.jit.marker import unified
# 检查GPU卡的可用性
paddle.utils.run_check()
```

如上代码执行成功，则认为环境可用。
