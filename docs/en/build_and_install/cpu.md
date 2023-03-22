English | [中文](../../cn/build_and_install/cpu.md)

# How to Build CPU Deployment Environment

## Build Options

Please do not modify other cmake paramters exclude the following options.

| Option                      | Supported Platform | Description                                                                        |
|:------------------------|:------- | :--------------------------------------------------------------------------|
| ENABLE_ORT_BACKEND      | Linux(x64/aarch64)/Windows(x64)/Mac OSX(arm64/x86) | Default OFF, whether to intergrate ONNX Runtime backend   |
| ENABLE_PADDLE_BACKEND   | Linux(x64)/Windows(x64) | Default OFF, whether to intergrate Paddle Inference backend             |  
| ENABLE_LITE_BACKEND   | Linux(aarch64) | Default OFF, whether to intergrate Paddle Lite backend             | 
| ENABLE_OPENVINO_BACKEND | Linux(x64)/Windows(x64)/Mac OSX(x86) | Default OFF, whether to intergrate OpenVINO backend      |
| ENABLE_VISION           | Linux(x64/aarch64)/Windows(x64)/Mac OSX(arm64/x86) | Default OFF, whether to intergrate vision models |
| ENABLE_TEXT             | Linux(x64/aarch64)/Windows(x64)/Mac OSX(arm64/x86) | Default OFF, whether to intergrate text models |
| WITH_CAPI             | Linux(x64)/Windows(x64)/Mac OSX(x86) | Default OFF, whether to intergrate C API  |
| WITH_CSHARPAPI        | Windows(x64) | Default OFF, whether to intergrate C# API   |

The configuration for third libraries(Optional, if the following option is not defined, the prebuilt third libraries will download automaticly while building FastDeploy).
| Option                     | Description                                                                                           |
| :---------------------- | :--------------------------------------------------------------------------------------------- |
| ORT_DIRECTORY           | While ENABLE_ORT_BACKEND=ON, use ORT_DIRECTORY to specify your own ONNX Runtime library path.  |
| OPENCV_DIRECTORY        | While ENABLE_VISION=ON, use OPENCV_DIRECTORY to specify your own OpenCV library path.     |
| OPENVINO_DIRECTORY      |  While ENABLE_OPENVINO_BACKEND=ON, use OPENVINO_DIRECTORY to specify your own OpenVINO library path.    |

## How to Build and Install C++ SDK

### Linux & Mac

Prerequisite for Compiling on Linux & Mac:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.18.0

It it recommend install OpenCV library manually, and define `-DOPENCV_DIRECTORY` to set path of OpenCV library(If the flag is not defined, a prebuilt OpenCV library will be downloaded automaticly while building FastDeploy, but the prebuilt OpenCV cannot support reading video file or other function e.g `imshow`)
```
sudo apt-get install libopencv-dev
```

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON \
         -DOPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4
make -j12
make install
```

### Windows

Prerequisite for Compiling on Windows:

- Windows 10/11 x64
- Visual Studio 2019

Launch the `x64 Native Tools Command Prompt for VS 2019` from the Windows Start Menu and run the following commands:

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
% nuget restore  （please execute it when WITH_CSHARPAPI=ON to prepare dependencies in C#)
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```

Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`

If you use CMake GUI, please refer to [How to Compile with CMakeGUI + Visual Studio 2019 IDE on Windows](../faq/build_on_win_with_gui.md)

## How to Build and Install Python SDK

Prerequisite for Compiling:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.18.0
- python >= 3.6

Notice the `wheel` is required if you need to pack a wheel, execute `pip install wheel` first.

All compilation options are introduced via environment variables

### Linux & Mac

It it recommend install OpenCV library manually, and define `-DOPENCV_DIRECTORY` to set path of OpenCV library(If the flag is not defined, a prebuilt OpenCV library will be downloaded automaticly while building FastDeploy, but the prebuilt OpenCV cannot support reading video file or other function e.g `imshow`)
```
sudo apt-get install libopencv-dev
```

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
# The OPENCV_DIRECTORY is optional, if not exported, a prebuilt OpenCV library will be downloaded
export OPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4

python setup.py build
python setup.py bdist_wheel
```

### Windows

Prerequisite for Compiling on Windows:

- Windows 10/11 x64
- Visual Studio 2019
- python >= 3.6

Launch the x64 Native Tools Command Prompt for VS 2019 from the Windows Start Menu and run the following commands:

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

The compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory once finished. Users can pip-install it directly.

During the compilation, if developers want to change the compilation parameters,  it is advisable to delete the `build` and `.setuptools-cmake-build` subdirectories in the `FastDeploy/python` to avoid the possible impact from cache, and then recompile.
