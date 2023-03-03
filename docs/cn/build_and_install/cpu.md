[English](../../en/build_and_install/cpu.md) | 简体中文

# CPU部署库编译

## 编译选项说明

无论是在何平台编译，编译时仅根据需求修改如下选项，勿修改其它参数
| 选项                      | 支持平台 | 说明                                                                        |
|:------------------------|:------- | :--------------------------------------------------------------------------|
| ENABLE_ORT_BACKEND      | Linux(x64/aarch64)/Windows(x64)/Mac OSX(arm64/x86) | 默认OFF, 是否编译集成ONNX Runtime后端    |
| ENABLE_PADDLE_BACKEND   | Linux(x64)/Windows(x64) | 默认OFF，是否编译集成Paddle Inference后端                             |  
| ENABLE_LITE_BACKEND   | Linux(aarch64) | 默认OFF，是否编译集成Paddle Lite后端                             |  
| ENABLE_OPENVINO_BACKEND | Linux(x64)/Windows(x64)/Mac OSX(x86) | 默认OFF，是否编译集成OpenVINO后端       |
| ENABLE_VISION           | Linux(x64)/Windows(x64)/Mac OSX(x86) |  默认OFF，是否编译集成视觉模型的部署模块                                                    |
| ENABLE_TEXT             | Linux(x64)/Windows(x64)/Mac OSX(x86) | 默认OFF，是否编译集成文本NLP模型的部署模块                                                  |
| WITH_CAPI             | Linux(x64)/Windows(x64)/Mac OSX(x86) | 默认OFF，是否编译集成C API  |
| WITH_CSHARPAPI        | Windows(x64) | 默认OFF，是否编译集成C# API  |

第三方库依赖指定（不设定如下参数，会自动下载预编译库）
| 选项                     | 说明                                                                                           |
| :---------------------- | :--------------------------------------------------------------------------------------------- |
| ORT_DIRECTORY           | 当开启ONNX Runtime后端时，用于指定用户本地的ONNX Runtime库路径；如果不指定，编译过程会自动下载ONNX Runtime库  |
| OPENCV_DIRECTORY        | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库              |
| OPENVINO_DIRECTORY      | 当开启OpenVINO后端时, 用于指定用户本地的OpenVINO库路径；如果不指定，编译过程会自动下载OpenVINO库             |

## C++ SDK编译安装

### Linux & Mac

Linux上编译需满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0

此外更推荐开发者自行安装，编译时通过`-DOPENCV_DIRECTORY`来指定环境中的OpenCV（如若不指定-DOPENCV_DIRECTORY，会自动下载FastDeploy提供的预编译的OpenCV，但在**Linux平台**无法支持Video的读取，以及imshow等可视化界面功能）
```
sudo apt-get install libopencv-dev
```

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON \
         -DOPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
         -DENABLE_TEXT=ON
make -j12
make install
```

### Windows

Windows编译需要满足条件

- Windows 10/11 x64
- Visual Studio 2019

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 ^
         -DENABLE_ORT_BACKEND=ON ^
         -DENABLE_PADDLE_BACKEND=ON ^
         -DENABLE_OPENVINO_BACKEND=ON ^
         -DENABLE_VISION=ON ^
         -DENABLE_TEXT=ON ^
         -DCMAKE_INSTALL_PREFIX="D:\Paddle\compiled_fastdeploy"
% nuget restore  （please execute it when WITH_CSHARPAPI=ON to prepare dependencies in C#)
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```

编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库

如您使用CMake GUI可参考文档[Windows使用CMakeGUI + Visual Studio 2019 IDE编译](../faq/build_on_win_with_gui.md)

## Python编译安装

编译过程同样需要满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0
- python >= 3.6

Python打包依赖`wheel`，编译前请先执行`pip install wheel`

所有编译选项通过环境变量导入

### Linux & Mac

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_TEXT=ON
# OPENCV_DIRECTORY可选，不指定会自动下载FastDeploy提供的预编译OpenCV库
export OPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4

python setup.py build
python setup.py bdist_wheel
```

### Windows

编译过程同样需要满足
- Windows 10/11 x64
- Visual Studio 2019
- python >= 3.6

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
set ENABLE_ORT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_OPENVINO_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_TEXT=ON

python setup.py build
python setup.py bdist_wheel
```

编译完成即会在`FastDeploy/python/dist`目录下生成编译后的`wheel`包，直接pip install即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除`FastDeploy/python`目录下的`build`和`.setuptools-cmake-build`两个子目录后再重新编译
