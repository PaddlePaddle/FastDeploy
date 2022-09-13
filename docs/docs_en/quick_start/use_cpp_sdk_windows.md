# Use FastDeploy C++ SDK on Windows


Using the FastDeploy C++ SDK on Windows is slightly different from using it on Linux. Below is an example of PPYOLOE for accelerated deployment on CPU/GPU, and GPU with TensorRT.

Two steps before deployment:

- 1. The hardware and software environment meets the requirements. Please refer to [FastDeploy Environment Requirements](../environment.md) for more details.
- 2. Download the pre-built deployment SDK and samples code according to the development environment. For more details, please refer to [install\_cpp\_sdk](./install_cpp_sdk.md)

## Dependencies

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (WITH_GPU=ON)
- cudnn >= 8.0 (WITH_GPU=ON)
- TensorRT >= 8.4 (ENABLE\_TRT\_BACKEND=ON)

## Download FastDeploy Windows 10 C++ SDK

Download a compiled FastDeploy Windows 10 C++ SDK from the link below. It contains the examples code.

```text
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.0.zip
```

## Prepare model files and test images

Download the model file and test images from the following link and unzip them

```text
https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz # (Unzip after download)
https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

## Compile PPYOLOE on Windows

Open the`x64 Native Tools Command Prompt for VS 2019`command tool on the Windows menu, and direct it to ppyoloe‘s demo path

```bat
cd fastdeploy-win-x64-gpu-0.2.0\examples\vision\detection\paddledetection\cpp
```

```bat
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.0 -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
```

Run the following command:

```bat
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

## Configure dependency library path

#### Method 1: Set environment variables on the command line

The compiled exe file is stored in the Release directory. Before running the demo, the model and test images need to be copied to this directory. Also, users need to designate the search path for the DLL in the terminal. Please execute the following command in the build directory.

```bat
set FASTDEPLOY_PATH=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.0
set PATH=%FASTDEPLOY_PATH%\lib;%FASTDEPLOY_PATH%\third_libs\install\onnxruntime\lib;%FASTDEPLOY_PATH%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin;%FASTDEPLOY_PATH%\third_libs\install\paddle_inference\paddle\lib;%FASTDEPLOY_PATH%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib;%FASTDEPLOY_PATH%\third_libs\install\paddle_inference\third_party\install\mklml\lib;%FASTDEPLOY_PATH%\third_libs\install\paddle2onnx\lib;%FASTDEPLOY_PATH%\third_libs\install\tensorrt\lib;%FASTDEPLOY_PATH%\third_libs\install\yaml-cpp\lib;%PATH%
```

Note: Copy onnxruntime.dll to the directory of the exe.

```bat
copy /Y %FASTDEPLOY_PATH%\third_libs\install\onnxruntime\lib\onnxruntime* Release\
```

As the latest Windows version contains onnxruntime.dll in the System32 system directory, there will still be a loading conflict for onnxruntime even if the PATH is set. To solve this problem, copy the onnxruntime.dll used in the demo to the directory of exe. Example is as follows:

```bat
where onnxruntime.dll
C:\Windows\System32\onnxruntime.dll  # windows自带的onnxruntime.dll
```

#### Method 2: Copy dependencies library to exe directory

Copy manually, or execute the following command in the build directory.

```bat
set FASTDEPLOY_PATH=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.0
copy /Y %FASTDEPLOY_PATH%\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\onnxruntime\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\paddle_inference\paddle\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\paddle_inference\third_party\install\mklml\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\paddle2onnx\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\tensorrt\lib\*.dll Release\
copy /Y %FASTDEPLOY_PATH%\third_libs\install\yaml-cpp\lib\*.dll Release\
```

## Run the demo

```bat
cd Release
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0  # CPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 1  # GPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 2  # GPU + TensorRT
```
