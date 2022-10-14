

# How to Build CPU Deployment Environment

FastDeploy currently supports the following backend engines on the CPU

| Backend               | Platform                                             | Supported model format | Description                                                                                      |
|:--------------------- |:---------------------------------------------------- |:---------------------- |:------------------------------------------------------------------------------------------------ |
| Paddle&nbsp;Inference | Windows(x64)<br>Linux(x64)                           | Paddle                 | The compilation switch `ENABLE_PADDLE_BACKEND` is controlled by ON or OFF. The default is OFF.   |
| ONNX&nbsp;Runtime     | Windows(x64)<br>Linux(x64/aarch64)<br>Mac(x86/arm64) | Paddle/ONNX            | The compilation switch `ENABLE_ORT_BACKEND` is controlled by ON or OFF. The default is OFF.      |
| OpenVINO              | Windows(x64)<br>Linux(x64)<br>Mac(x86)               | Paddle/ONNX            | The compilation switch `ENABLE_OPENVINO_BACKEND` is controlled by ON or OFF. The default is OFF. |

## How to Build and Install C++ SDK

### Linux & Mac

Prerequisite for Compiling on Linux:

- gcc/g++ >= 5.4 (8.2 is recommended)
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

Prerequisite for Compiling on Windows: 

- Windows 10/11 x64
- Visual Studio 2019

Open the `x64 Native Tools Command Prompt for VS 2019` in the windows menu and run the following commands: 

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

Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`

If you use CMake GUI, please refer to [How to Compile with CMakeGUI + Visual Studio 2019 IDE on Windows](../faq/build_on_win_with_gui.md)

## How to Build and Install Python SDK

Prerequisite for Compiling: 

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.18.0
- python >= 3.6

All compilation options are introduced via environment variables

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

Prerequisite for Compiling:

- Windows 10/11 x64
- Visual Studio 2019
- python >= 3.6

Open the `x64 Native Tools Command Prompt for VS 2019` in the windows menu and run the following commands:

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
