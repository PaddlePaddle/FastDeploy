# CPU部署环境

FastDeploy当前在CPU支持后端引擎如下

| 后端 | 平台  | 支持模型格式 | 说明 |
| :--- | :---- | :----------- | :--- |
| Paddle&nbsp;Inference | Windows(x64)<br>Linux(x64) | Paddle | 编译开关`ENABLE_PADDLE_BACKEND`为ON或OFF控制, 默认OFF |
| ONNX&nbsp;Runtime | Windows(x64)<br>Linux(x64/aarch64)<br>Mac(x86/arm64) | Paddle/ONNX | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF |
| OpenVINO | Windows(x64)<br>Linux(x64) | Paddle/ONNX | 编译开关`ENABLE_OPENVINO_BACKEND`为ON或OFF控制，默认OFF |

## 预编译库安装

FastDeploy提供了预编译库供开发者快速安装使用，默认集成了各推理后端及Vision和Text模块, 当前发布两种版本

- Release版本：FastDeploy每月更新发布的已测试版本
- Nightly build版本：FastDeploy每日定期根据最新代码发布的编译版本(仅含Linux-x64和Windows-x64版本)

### Python安装

Release版本安装
```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Nightly build版本安装
```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

### C++SDK安装

注：其中`nightly build`为每日最新代码编译产出

| 平台  | 下载链接(Release) | 下载链接(nightly build) | 说明 |
| :---- | :---------------- | :---------------------- | :--- |
| Linux x64 | [fastdeploy-linux-x64-0.2.1.tgz]() | [fastdeploy-linux-x64-0.2.2-dev.tgz]() | gcc 8.2编译产出 |
| Linux aarch64 | [fastdeploy-linux-x64-0.2.1.tgz]() | - | gcc 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-0.2.1.zip]() | [fastdeploy-win-x64-0.2.2.-dev.tgz]() | Visual Studio 2019编译产出 |
| Mac x86 | [fastdeploy-osx-x86_64-0.2.1.tgz]() | - | OSX 10.0编译产出，仅含ONNX Runtime后端 |
| Max arm64 | [fastdeploy-osx-arm64-0.2.1.tgz]() | - | OSX 11.0编译产出，仅含ONNX Runtime后端 |

## C++ SDK编译安装

### Linux & Mac

Linux上编译需满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON
make -j12
make install
```

### Windows

Windows编译需要满足条件

- Windows 10/11 x64
- Visual Studio 2019

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 \
         -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DENABLE_VISION=ON \
         -DCMAKE_INSTALL_PREFIX="D:\Paddle\compiled_fastdeploy"
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```

编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库


## Python包编译安装

编译过程同样需要满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0
- python >= 3.6

所有编译选项通过环境变量导入

### Linux & Mac

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel
```

### Windows

编译过程同样需要满足
- Windows 10/11 x64
- Visual Studio 2019
- python >= 3.6

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
set ENABLE_ORT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_OPENVINO_BACKEND=ON
set ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel
```

编译完成即会在`FastDeploy/python/dist`目录下生成编译后的`wheel`包，直接pip install即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除`FastDeploy/python`目录下的`build`和`.setuptools-cmake-build`两个子目录后再重新编译
