[简体中文](../../zh/get_started/installation/nvidia_gpu.md)

# NVIDIA CUDA GPU Installation

The following installation methods are available when your environment meets these requirements:

- GPU Driver >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Python >= 3.10
- Linux X86_64

## 1. Pre-built Docker Installation (Recommended)

**Notice**: The pre-built image only supports SM80/90 GPU(e.g. H800/A800)，if you are deploying on SM86/89GPU(L40/4090/L20), please reinstall ```fastdeploy-gpu``` after you create the container.

```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.3.0
```

## 2. Pre-built Pip Installation

First install paddlepaddle-gpu. For detailed instructions, refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
```shell
# Install stable release
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install latest Nightly build
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
```

Then install fastdeploy. **Do not install from PyPI**. Use the following methods instead:

For SM80/90 architecture GPUs(e.g A30/A100/H100/):
```
# Install stable release
python -m pip install fastdeploy-gpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install latest Nightly build
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

For SM86/89 architecture GPUs(e.g A10/4090/L20/L40):
```
# Install stable release
python -m pip install fastdeploy-gpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

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
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
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

## 5. Precompiled Operator Wheel Packages

FastDeploy provides precompiled GPU operator wheel packages for quick setup without building the entire source code.
This method currently supports **SM90 architecture (e.g., H20/H100)** and **CUDA 12.6** environments only.

> By default, `build.sh` compiles all custom operators from source.To use the precompiled package, enable it with the `FD_USE_PRECOMPILED` parameter.
> If the precompiled package cannot be downloaded or does not match the current environment, the system will automatically fall back to `4. Build Wheel from Source`.

First, install paddlepaddle-gpu.
For detailed instructions, please refer to the [PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/).

```shell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Then, clone the FastDeploy repository and build using the precompiled operator wheels:

```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

# Argument 1: Whether to build wheel package (1 for yes)
# Argument 2: Python interpreter path
# Argument 3: Whether to compile CPU inference operators (false for GPU only)
# Argument 4: Target GPU architectures (currently supports [90])
# Argument 5: Whether to use precompiled operators (1 for enable)
# Argument 6 (optional): Specific commitID for precompiled operators(The default is the current commit ID.)

# Use precompiled operators for accelerated build
bash build.sh 1 python false [90] 1

# Use precompiled wheel from a specific commit
bash build.sh 1 python false [90] 1 8a9e7b53af4a98583cab65e4b44e3265a93e56d2
```

The downloaded wheel packages will be stored in the `FastDeploy/pre_wheel` directory.
After the build completes, the operator binaries can be found in `FastDeploy/fastdeploy/model_executor/ops/gpu`.

> **Notes:**
>
> - This mode prioritizes downloading precompiled GPU operator wheels to reduce build time.
> - Currently supports **GPU + SM90 + CUDA 12.6** only.
> - For custom architectures or modified operator logic, please use **source compilation (Section 4)**.
> - You can check whether the precompiled wheel for a specific commit has been successfully built on the [FastDeploy CI Build Status Page](https://github.com/PaddlePaddle/FastDeploy/actions/workflows/ci_image_update.yml).

## Environment Verification

After installation, verify the environment with this Python code:
```python
import paddle
from paddle.jit.marker import unified
# Verify GPU availability
paddle.utils.run_check()
```
If the above code executes successfully, the environment is ready.
