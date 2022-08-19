# 在 Windows 使用 FastDeploy C++ SDK

在 Windows 下使用 FastDeploy C++ SDK 与在 Linux 下使用稍有不同。以下以 PPYOLOE 为例进行演示在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/the%20software%20and%20hardware%20requirements.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

## 环境依赖

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 11.2 (当WITH_GPU=ON)
- TensorRT >= 8.4 (当ENABLE_TRT_BACKEND=ON)

## 下载 FastDeploy Windows 10 C++ SDK
可以从以下链接下载编译好的 FastDeploy Windows 10 C++ SDK，SDK中包含了examples代码。
```text
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.0.zip
```
## 准备模型文件和测试图片  
可以从以下链接下载模型文件和测试图片，并解压缩
```text
https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz # (下载后解压缩)
https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

## 在 Windows 上编译 PPYOLOE
Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，cd到ppyoloe的demo路径  
```bat  
cd fastdeploy-win-x64-gpu-0.2.0\examples\vision\detection\paddledetection\cpp
```
```bat
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.0 -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
```
然后执行
```bat
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```
编译好的exe保存在Release目录下，在运行demo前，需要将模型和测试图片拷贝至该目录。另外，需要在终端指定DLL的搜索路径。请在build目录下执行以下命令。
```bat
set FASTDEPLOY_PATH=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.0
set PATH=%FASTDEPLOY_PATH%\lib;%FASTDEPLOY_PATH%\third_libs\install\onnxruntime\lib;%FASTDEPLOY_PATH%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin;%FASTDEPLOY_PATH%\third_libs\install\paddle_inference\paddle\lib;%FASTDEPLOY_PATH%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib;%FASTDEPLOY_PATH%\third_libs\install\paddle_inference\third_party\install\mklml\lib;%FASTDEPLOY_PATH%\third_libs\install\paddle2onnx\lib;%FASTDEPLOY_PATH%\third_libs\install\tensorrt\lib;%FASTDEPLOY_PATH%\third_libs\install\yaml-cpp\lib;%PATH%
```
注意，需要拷贝onnxruntime.dll到exe所在的目录。
```bat
copy /Y %FASTDEPLOY_PATH%\third_libs\install\onnxruntime\lib\onnxruntime* Release\
```
由于较新的Windows在System32系统目录下自带了onnxruntime.dll，因此就算设置了PATH，系统依然会出现onnxruntime的加载冲突。因此需要先拷贝demo用到的onnxruntime.dll到exe所在的目录。
```bat
where onnxruntime.dll
C:\Windows\System32\onnxruntime.dll  # windows自带的onnxruntime.dll
```
## 运行 demo
```bat
cd Release
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0  # CPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 1  # GPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 2  # GPU + TensorRT
```
