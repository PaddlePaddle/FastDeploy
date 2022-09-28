# GPU部署环境

FastDeploy当前在GPU环境支持Paddle Inference、ONNX Runtime和TensorRT，但同时在Linux&Windows的GPU环境也同时支持CPU硬件，因此编译时也可以同步将CPU的推理后端OpenVINO编译集成

| 后端 | 平台  | 支持模型格式 | 说明 |
| :--- | :---- | :----------- | :--- |
| Paddle&nbsp;Inference | Windows(x64)<br>Linux(x64) | Paddle | 同时支持CPU/GPU，编译开关`ENABLE_PADDLE_BACKEND`为ON或OFF控制, 默认OFF |
| ONNX&nbsp;Runtime | Windows(x64)<br>Linux(x64/aarch64)<br>Mac(x86/arm64) | Paddle/ONNX | 同时支持CPU/GPU，编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF |
| TensorRT | Windows(x64)<br>Linux(x64) | Paddle/ONNX | 仅支持GPU，编译开关`ENABLE_TRT_BACKEND`为ON或OFF控制，默认OFF |
| OpenVINO | Windows(x64)<br>Linux(x64) | Paddle/ONNX | 仅支持CPU，编译开关`ENABLE_OPENVINO_BACKEND`为ON或OFF控制，默认OFF |

注意编译GPU环境时，需额外指定`WITH_GPU`为ON，设定`CUDA_DIRECTORY`，如若需集成TensorRT，还需同时设定`TRT_DIRECTORY`

## 预编译库安装

FastDeploy提供了预编译库供开发者快速安装使用，默认集成了各推理后端及Vision和Text模块, 当前发布两种版本

- Release版本：FastDeploy每月更新发布的已测试版本
- Nightly build版本：FastDeploy每日定期根据最新代码发布的编译版本(仅含Linux-x64和Windows-x64版本)

### Python安装

Release版本安装
```
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Nightly build版本安装
```
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

### C++SDK安装

注：其中`nightly build`为每日最新代码编译产出

| 平台  | 下载链接(Release) | 下载链接(nightly build) | 说明 |
| :---- | :---------------- | :---------------------- | :--- |
| Linux x64 | [fastdeploy-linux-x64-gpu-0.2.1.tgz]() | [fastdeploy-linux-x64-gpu-0.2.2-dev.tgz]() | gcc 8.2编译产出，CUDA 11.2，CUDNN 8.2 |
| Windows x64 | [fastdeploy-win-x64-gpu-0.2.1.zip]() | [fastdeploy-win-x64-gpu-0.2.2.-dev.tgz]() | Visual Studio 2019编译产出，CUDA 11.2，CUDNN 8.2 |

## C++ SDK编译安装

### Linux 

Linux上编译需满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0
- cuda >= 11.2
- cudnn >= 8.2

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DENABLE_TRT_BACKEND=ON \
         -DWITH_GPU=ON \
         -DTRT_DIRECTORY=/Paddle/TensorRT-8.4.1.5 \
         -DCUDA_DIRECTORY=/usr/local/cuda \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON
make -j12
make install
```

### Windows

Windows编译需要满足条件

- Windows 10/11 x64
- Visual Studio 2019
- cuda >= 11.2
- cudnn >= 8.2

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 \
         -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DENABLE_TRT_BACKEND=ON
         -DENABLE_VISION=ON \
         -DWITH_GPU=ON \
         -DTRT_DIRECTORY="D:\Paddle\TensorRT-8.4.1.5" \
         -DCUDA_DIRECTORY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2" \
         -DCMAKE_INSTALL_PREFIX="D:\Paddle\compiled_fastdeploy"
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```

编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库

## Python包编译安装


### Linux

编译过程需要满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0
- python >= 3.6
- cuda >= 11.2
- cudnn >= 8.2

所有编译选项通过环境变量导入

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export TRT_DIRECTORY=/Paddle/TensorRT-8.4.1.5
export CUDA_DIRECTORY=/usr/local/cuda

python setup.py build
python setup.py bdist_wheel
```

### Windows

编译过程同样需要满足
- Windows 10/11 x64
- Visual Studio 2019
- python >= 3.6
- cuda >= 11.2
- cudnn >= 8.2

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export TRT_DIRECTORY="D:\Paddle\TensorRT-8.4.1.5"
export CUDA_DIRECTORY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"

python setup.py build
python setup.py bdist_wheel
```

编译完成即会在`FastDeploy/python/dist`目录下生成编译后的`wheel`包，直接pip install即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除`FastDeploy/python`目录下的`build`和`.setuptools-cmake-build`两个子目录后再重新编译
