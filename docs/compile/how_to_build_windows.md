# Windows编译

## 环境依赖

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 8.0 (当WITH_GPU=ON)
- TensorRT >= 8.4 (当ENABLE_TRT_BACKEND=ON)

## 编译CPU版本 C++ SDK

Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，其中`CMAKE_INSTALL_PREFIX`用于指定编译后生成的SDK路径

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64 -DENABLE_ORT_BACKEND=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_VISION_VISUALIZE=ON
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```
编译后，FastDeploy CPU C++ SDK即在`D:\Paddle\FastDeploy\build\fastdeploy-win-x64`目录下

## 编译GPU版本 C++ SDK

Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，其中`CMAKE_INSTALL_PREFIX`用于指定编译后生成的SDK路径

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64-gpu -DWITH_GPU=ON -DENABLE_ORT_BACKEND=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_VISION_VISUALIZE=ON -DCUDA_DIRECTORY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64  

% 附加说明：%
% (1) -DCUDA_DIRECTORY指定CUDA所在的目录 %
% (2) 若编译Paddle后端，设置-DENABLE_PADDLE_BACKEND=ON %
% (3) 若编译TensorRT后端，需要设置-DENABLE_TRT_BACKEND=ON，并指定TRT_DIRECTORY %
% (4) 如-DTRT_DIRECTORY=D:\x64\third_party\TensorRT-8.4.1.5 %
```
编译后，FastDeploy GPU C++ SDK即在`D:\Paddle\FastDeploy\build\fastdeploy-win-x64-gpu`目录下

## 编译CPU版本 Python Wheel包

Windows菜单打开x64 Native Tools Command Prompt for VS 2019命令工具。Python编译时，通过环境变量获取编译选项，在命令行终端运行以下命令
```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python && git checkout develop

set ENABLE_ORT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_VISION_VISUALIZE=ON

% 这里指定用户自己的python解释器 以python3.8为例 %
C:\Python38\python.exe setup.py build
C:\Python38\python.exe setup.py bdist_wheel
```
编译好的wheel文件在dist目录下，pip安装编译好的wheel包，命令如下
```bat
C:\Python38\python.exe -m pip install dist\fastdeploy_python-0.2.1-cp38-cp38-win_amd64.whl
```

## 编译GPU版本 Python Wheel包  
Windows菜单打开x64 Native Tools Command Prompt for VS 2019命令工具。Python编译时，通过环境变量获取编译选项，在命令行终端运行以下命令
```bat
% 说明：CUDA_DIRECTORY 为用户自己的CUDA目录 以下为示例 %
set CUDA_DIRECTORY=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
% 说明：TRT_DIRECTORY 为下载的TensorRT库所在的目录 以下为示例 如果不编译TensorRT后端 可以不设置 %
set TRT_DIRECTORY=D:\x64\third_party\TensorRT-8.4.1.5
set WITH_GPU=ON
set ENABLE_ORT_BACKEND=ON
% 说明：如果不编译TensorRT后端 此项为OFF %
set ENABLE_TRT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_VISION_VISUALIZE=ON

git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop

% 说明：这里指定用户自己的python解释器 以python3.8为例 %
C:\Python38\python.exe setup.py build
C:\Python38\python.exe setup.py bdist_wheel
```
编译好的wheel文件在dist目录下，pip安装编译好的wheel包，命令如下
```bat
C:\Python38\python.exe -m pip install dist\fastdeploy_gpu_python-0.2.1-cp38-cp38-win_amd64.whl
```
更多编译选项说明参考[编译指南](./README.md)
