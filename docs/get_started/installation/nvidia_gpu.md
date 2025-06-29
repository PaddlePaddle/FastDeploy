# NVIDIA CUDA GPU Installation

The following installation methods are available when your environment meets these requirements:

- GPU Driver >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Python >= 3.10
- Linux X86_64

## 1. Pre-built Docker Installation (Recommended)
```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy:${fastdeploy_latest_version}
```
Where ```${fastdeploy_latest_version}``` is the FastDeploy release version number. [Check latest release here](https://github.com/PaddlePaddle/FastDeploy/releases). For example:
```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy:2.0.0
```

## 2. Pre-built Pip Installation

First install paddlepaddle-gpu. For detailed instructions, refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
```shell
python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Then install fastdeploy. **Do not install from PyPI**. Use the following methods instead:

For SM80/90 architecture GPUs(e.g A100/H100):
```
# Install stable release
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install latest Nightly build
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

For SM86/89 architecture GPUs(e.g 4090/L20/L40):
```
# Install stable release
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install latest Nightly build
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 3. Build from Source Using Docker

- Note: ```dockerfiles/Dockerfile.gpu``` by default supports SM 80/90 architectures. To support other architectures, modify ```bash build.sh 1 python false [80,90]``` in the Dockerfile. It's recommended to specify no more than 2 architectures.

```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

docker build -f dockerfiles/Dockerfile.gpu -t fastdeploy:gpu .
```

## 4. Build Wheel from Source

First install paddlepaddle-gpu. For detailed instructions, refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
```shell
python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Then clone the source code and build:
```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

# Argument 1: Whether to build wheel package (1 for yes, 0 for compile only)
# Argument 2: Python interpreter path
# Argument 3: Whether to compile CPU inference operators
# Argument 4: Target GPU architectures
bash build.sh 1 python false [80,90]
```
The built packages will be in the ```FastDeploy/dist``` directory.

## Environment Verification

After installation, verify the environment with this Python code:
```python
import paddle
from paddle.jit.marker import unified
# Verify GPU availability
paddle.utils.run_check()
# Verify FastDeploy custom operators compilation
from fastdeploy.model_executor.ops.gpu import beam_search_softmax
```
If the above code executes successfully, the environment is ready.
