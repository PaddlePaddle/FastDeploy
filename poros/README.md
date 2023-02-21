# Poros AI Compiler

## Description

Poros is an AI Compiler for deep learning framework. It can provide significantly lower inference latency comparing with original model, and provide much flexibility for dynamic graphs.
Poros mainly works on the TorchScript IR currently, that means it supports the models from PyTorch, ONNX, TensorFlow and any other framework that can be converted to TorchScript. Also, we are planning to support more IRs in the future.
Poros is designed to supports multiple hardware backends conveniently. For now, Poros has supported GPU and XPU (BAIDU-Kunlun) Device. It's welcomed to add additional devices.

## How It Works

Figure 1 is the architecture of Poros. The central part marked by the red dotted line is Model Optimizer, the main module of Poros. IR graphs are optimized by IR lowering, op fusing, op converting and auto-tuning, and then segmented into engine related subgraph by maximize the op nums of each engine kernel and minimize the total count of engine kernels.

![image](https://user-images.githubusercontent.com/54064850/203691621-e75d7c17-320c-4dff-8abe-58c3c9db99a2.png)

In order to achieve the above goals on GPU, we've rewritten hundreds of TorchScript OPs, which reduced extra subgraphs caused by unsupported op during subgraph partitioning. Dozens of lowering strategy including op fusions were employed to reduce the actual calculating load of CUDA Kernels.

## Dependencies

Poros is developed based on PyTorch, CUDA, TensorRT (TRT Engine), CuDNN. The minimum_required (recommended) versions of
these packages are listed as below:

| Package  | Minimum Version | Recommended Version |
|----------|-----------------|---------------------|
| PyTorch  | 1.9.0           | 1.12.1              |
| CUDA     | 10.2            | 11.3                |
| TensorRT | 8.2             | 8.4                 |
| CuDNN    | 7.6.5           | 8.4                 |
| Python   | 3.6.5           | 3.8                 |

If you want to build for GPU Inference, it's better to align the CUDA version with the version that PyTorch built on.
For example, we recommend you to use CUDA 11.1+ if the installed PyTorch version is 1.11.0+cu111, or some "undefined
reference CUDA...." errors may appear during building.

> There is a known cuBlas related issue of CUDA 10.2. If you are using CUDA 10.2, make sure these two patches have be installed. 
> https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

## How To Build

### 0. Install Dependencies

get Poros source code:

```shell
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd poros
git submodule update --init --recursive
```

We strongly recommend you to prepare the building environment with anaconda3:

```shell
conda create --name poros python=3.8
conda activate poros
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
conda install cmake==3.22.1 pytorch==1.12.1 cudatoolkit=11.3 numpy -c pytorch
```
**If CUDA has been installed as system driver, cudatoolkit is not necessary. And CMake version requires >= 3.21, GCC version requires >= 8.2.**


Poros uses cmake to manage dependencies. It will find all dependency packages automatically as long as the packages were
installed to the usual location. Otherwise, you should assign the install location of these packages manually.

```shell
export CUDAToolkit_ROOT=/cuda/install/dir/  #point CUDAToolkit_ROOT to the CUDA installation dir
export TENSORRT_ROOT=/tensorrt/install/dir/ #download from Nvidia and upack, no need to install into system
export CUDNN_ROOT=/cudnn/install/dir/       #download from Nvidia and upack, no need to install into system
```
Add cuda, tensorrt and cudnn into your environment variables.

```shell
export PATH=$CUDAToolkit_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAToolkit_ROOT/lib64:$TENSORRT_ROOT/lib:$CUDNN_ROOT/lib:$LD_LIBRARY_PATH
```

Additional dependency `mkl` is needed while building with PyTorch1.11 + CUDA11.1
It can be added into cmake by installing, if not, you can try to add it by:
```shell
conda install mkl
```

Other packages that Poros depend on are: gflags, googletest etc. , they can be downloaded
by ` git submodule update --init --recursive --jobs 0 -f`

### 1. Build Project with CMake

```shell
cd poros
mkdir build
cd build
cmake ..
make 
```

By default, only the shared library (libporos.so) will be built.

**To build a static lib (libporos.a):**

```shell
cmake -DBUILD_STATIC=on ..
make 
```

Poros `kernel` contains the framework of Poros, as well as the IR lowering strategy, the sub-graph segmentation strategy
and the engine manager without any specific engine (e.g. TensorRT). For Developers who want to use their own
engines, `kernel` can be built separately with options as below:

**To build a shared kernel lib (libporos-kernel.so):**

```shell
cmake -DBUILD_KERNEL=on ..
make 
```

**To build a static kernel lib (libporos-kernel.a):**

```shell
cmake -DBUILD_STATIC_KERNEL=on ..
make 
```

### 2. Build Distributing Package with setuptools (Python3)

After the libporos.so has been built, you can build the `.whl` package for Python3:

```shell
cd ../python
python3 setup.py bdist_wheel
```

The output looks like: `poros-0.1.0-cp38-cp38m-linux_x86_64.whl`. It can be installed easily with pip:

```shell
cd dist
pip3 install poros-0.1.0-cp38-cp38m-linux_x86_64.whl
```
or, you can use `python3 setup.py develop` to create symbolic link to `python` dir.

### 3. Build Executable Binary

We provide an example C++ shell for users who want to build an executable binary. The `main.cpp` file locates
at `tools/main.cpp`, you modify the code according to your needs. The executable binary `poros-tool` can be built with
this command:

```shell
mkdir build
cd build
cmake -DBUILD_TOOL=on ..
make 
```

### 4. Build Test
```shell
cmake -DUT=on ..
make 
./unit_test # run unit test
```


## How To Use

### 1. Python Usage:

```python
import poros
import torch
from torchvision import models

original_model = models.resnet50(pretrained=True).cuda().eval() #load/download pre-trained model
option = poros.PorosOptions() #set poros option
poros_model = poros.compile(torch.jit.script(original_model), input_datas, option) #build the model

input = torch.randn(1,3,224,224, dtype=torch.float32).cuda()
poros_res = poros_model(input) # use compiled model in the same way as the original model

```

The complete benchmark example (resnet50) .py script is `python/example/test_resnet.py`

```shell
python3 python/example/test_resnet.py
```

### 2. CPP Usage:

If the executable binary `poros-tool` is built, you can run the benchmark like this:

```shell
./poros-tool --module_file_path ../../poros/tools/std_pretrained_resnet50_gpu.pt --test_mode=original #original PyTorch model
./poros-tool --module_file_path ../../poros/tools/std_pretrained_resnet50_gpu.pt --test_mode=poros #poros compiled model
```
> PyTorch has changed the packaging format of model since 1.4+, while the pretrained model of resnet50 is still using the old format (.tar).
> You may need to convert the format to the newer one (.zip) by your self. Convert command like this:
> ```python
> original_model = models.resnet50(pretrained=True).cuda().eval()
> torch.save(original_model, 'std_pretrained_resnet50_gpu.pt', _use_new_zipfile_serialization=False)
> ```

## Benchmark

Take a look at the [Benchmark](docs/Benchmark.md).

## Acknowledgement
Poros has been incubated for more than 2 years. In this project, NVIDIA helped us a lot (especially  Gary Ji, Vincent Zhang, Jie Fang). They answered lots of technical questions about GPU and gave us many suggestions. Appreciate their great support.
