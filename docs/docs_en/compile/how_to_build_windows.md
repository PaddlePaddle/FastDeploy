# Compile on Windows

## Dependencies

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (WITH_GPU=ON)
- cudnn >= 8.0 (WITH_GPU=ON)
- TensorRT >= 8.4 (ENABLE_TRT_BACKEND=ON)

## Compile C++ SDK for CPU

Opens the `x64 Native Tools Command Prompt for VS 2019` command tool on the Windows menu. In particular, the `CMAKE_INSTALL_PREFIX` is used to designate the path to the SDK generated after compilation.

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64-0.2.1-DENABLE_ORT_BACKEND=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_VISION_VISUALIZE=ON
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```

After compilation, the FastDeploy CPU C++ SDK is in the `D:\Paddle\FastDeploy\build\fastdeploy-win-x64-0.2.1` directory

## Compile C++ SDK for GPU

Opens the `x64 Native Tools Command Prompt for VS 2019` command tool on the Windows menu. In particular, the `CMAKE_INSTALL_PREFIX` is used to designate the path to the SDK generated after compilation.

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64-gpu-0.2.1 -DWITH_GPU=ON -DENABLE_ORT_BACKEND=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_VISION_VISUALIZE=ON -DCUDA_DIRECTORY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64  

% Notes：%
% (1) -DCUDA_DIRECTORY designates the directory of CUDA %
% (2) If compile the Paddle backend, set-DENABLE_PADDLE_BACKEND=ON %
% (3) If compile the TensorRT backend，set-DENABLE_TRT_BACKEND=ON, and designate TRT_DIRECTORY %
% (4) If-DTRT_DIRECTORY=D:\x64\third_party\TensorRT-8.4.1.5 %
```

After compilation, FastDeploy GPU C++ SDK is under`D:\Paddle\FastDeploy\build\fastdeploy-win-x64-gpu-0.2.0`

## Compile Python Wheel package for CPU

Opens the `x64 Native Tools Command Prompt for VS 2019` command tool on the Windows menu. In particular, the compilation options are obtained through the environment variables and run the following command in terminal.

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop

set ENABLE_ORT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_VISION_VISUALIZE=ON

% 这里指定用户自己的python解释器 以python3.8为例 %
C:\Python38\python.exe setup.py build
C:\Python38\python.exe setup.py bdist_wheel
```

The compiled wheel files are in the dist directory. Use pip to install the compiled wheel package with the following command:

```bat
C:\Python38\python.exe -m pip install dist\fastdeploy_python-0.2.0-cp38-cp38-win_amd64.whl
```

## Compile Python Wheel package for GPU

Opens the `x64 Native Tools Command Prompt for VS 2019` command tool on the Windows menu. In particular, the compilation options are obtained through the environment variables and run the following command in terminal.

```bat
% Note：CUDA_DIRECTORY is your own CUDA directory. The following is an example %
set CUDA_DIRECTORY=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
% Note：TRT_DIRECTORY is the directory of the downloaded TensorRT library. The following is an example. Ignore the setting if the TensorRT backend is not needed. %
set TRT_DIRECTORY=D:\x64\third_party\TensorRT-8.4.1.5
set WITH_GPU=ON
set ENABLE_ORT_BACKEND=ON
% Note：If not compile TensorRT backend, the default is OFF %
set ENABLE_TRT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_VISION_VISUALIZE=ON

git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop

% Note: Designate your own python interpreter here. Take python 3.8 as an example %
C:\Python38\python.exe setup.py build
C:\Python38\python.exe setup.py bdist_wheel
```

The compiled wheel files are in the dist directory. Use pip to install the compiled wheel package with the following command:

```bat
C:\Python38\python.exe -m pip install dist\fastdeploy_gpu_python-0.2.0-cp38-cp38-win_amd64.whl
```

For more details, please refer to [Compile Readme](./README.md)
