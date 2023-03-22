English | [简体中文](README_CN.md)
# YOLOv5 C# Deployment Example

This directory provides `infer.cs` to finish the deployment of YOLOv5 on CPU/GPU.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Please follow below instructions to compile and test in Windows. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4) is required to support this model.

## 1. Download C# package management tool nuget client
> https://dist.nuget.org/win-x86-commandline/v6.4.0/nuget.exe

Add nuget program into system variable **PATH**

## 2. Download model and image for test
> https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx
> https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

## 3. Compile example code

Open `x64 Native Tools Command Prompt for VS 2019` command tool on Windows, cd to the demo path of ppyoloe and execute commands

```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\detection\yolov5\csharp

mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=D:\Download\fastdeploy-win-x64-gpu-x.x.x -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"

nuget restore
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

For more information about how to use FastDeploy SDK to compile a project with Visual Studio 2019. Please refer to
- [Using the FastDeploy C++ SDK on Windows Platform](../../../../../docs/en/faq/use_sdk_on_windows.md)

## 4. Execute compiled program

fastdeploy.dll and related dynamic libraries are required by the program. FastDeploy provide a script to copy all required dll to your program path.

```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x

fastdeploy_init.bat install %cd% D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\detection\yolov5\csharp\build\Release
```

Then you can run your program and test the model with image
```shell
cd Release
infer_demo yolov5s.onnx 000000014439.jpg 0 # CPU
infer_demo yolov5s.onnx 000000014439.jpg 1  # GPU
```

## YOLOv5 C# Interface

### Model Class

```c#
fastdeploy.vision.detection.YOLOv5(
        string model_file,
        string params_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.ONNX)
```

> YOLOv5 initialization.

> **Params**

>> * **model_file**(str): Model file path
>> * **params_file**(str): Parameter file path，when model format is onnx，this can be empty string
>> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
>> * **model_format**(ModelFormat): Model format.


#### Predict Function

```c#
fastdeploy.DetectionResult Predict(OpenCvSharp.Mat im)
```

> Model prediction interface. Input images and output results directly.
>
> **Params**
>
>> * **im**(Mat): Input images in HWC or BGR format
>
> **Return**
>
>> * **result**(DetectionResult): Detection result, including detection box and confidence of each box. Refer to [Vision Model Prediction Result](../../../../../docs/api/vision_results/) for DetectionResult


- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
