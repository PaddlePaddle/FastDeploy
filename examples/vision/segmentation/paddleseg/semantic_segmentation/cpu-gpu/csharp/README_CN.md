[English](README.md) | 简体中文
# PaddleSeg CPU-GPU C#部署示例

本目录下提供`infer.cs`来调用C# API快速完成PaddleSeg模型在CPU/GPU上部署的示例。

## 1. 说明  
PaddleSeg支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署Segmentation模型。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleSeg部署模型](../README.md)，如果你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../matting)。


在本目录执行如下命令即可在Windows完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

## 4. 下载C#包管理程序nuget客户端
> https://dist.nuget.org/win-x86-commandline/v6.4.0/nuget.exe

下载完成后将该程序添加到环境变量**PATH**中

## 4. 下载模型文件和测试图片
> https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz # (下载后解压缩)
> https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png


## 6. 编译示例代码

本文档编译的示例代码可在解压的库中找到，编译工具依赖VS 2019的安装，**Windows打开x64 Native Tools Command Prompt for VS 2019命令工具**，通过如下命令开始编译

```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\segmentation\paddleseg\cpu-gpu\csharp

mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=D:\Download\fastdeploy-win-x64-gpu-x.x.x -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"

nuget restore
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

关于使用Visual Studio 2019创建sln工程，或者CMake工程等方式编译的更详细信息，可参考如下文档
- [在 Windows 使用 FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)
- [FastDeploy C++库在Windows上的多种使用方式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows_build.md)

## 7. 运行可执行程序

注意Windows上运行时，需要将FastDeploy依赖的库拷贝至可执行程序所在目录, 或者配置环境变量。FastDeploy提供了工具帮助我们快速将所有依赖库拷贝至可执行程序所在目录,通过如下命令将所有依赖的dll文件拷贝至可执行程序所在的目录(可能生成的可执行文件在Release下还有一层目录，这里假设生成的可执行文件在Release处)
```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x

fastdeploy_init.bat install %cd% D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\segmentation\paddleseg\cpu-gpu\csharp\build\Release
```

将dll拷贝到当前路径后，准备好模型和图片，使用如下命令运行可执行程序即可
```shell
cd Release
# CPU推理
infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 0
# GPU推理
infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 1
```

## 8. PaddleSeg C#接口

### 模型

```c#
fastdeploy.vision.segmentation.PaddleSeg(
        string model_file,
        string params_file,
        string config_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> PaddleSeg模型加载和初始化。

> **参数**

>> * **model_file**(str): 模型文件路径
>> * **params_file**(str): 参数文件路径
>> * **config_file**(str): 配置文件路径
>> * **runtime_option**(RuntimeOption): 后端推理配置，默认为null，即采用默认配置
>> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

#### Predict函数

```c#
fastdeploy.SegmentationResult Predict(OpenCvSharp.Mat im)
```

> 模型预测接口，输入图像直接输出结果。
>
> **参数**
>
>> * **im**(Mat): 输入图像，注意需为HWC，BGR格式
>>
> **返回值**
>
>> * **result**: Segmentation检测结果，SegmentationResult说明参考[视觉模型预测结果](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/)

## 9. 常见问题  

- [PPSegmentation 系列模型介绍](../../)
- [PaddleSeg Python部署](../python)
- [模型预测结果说明](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/)
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
