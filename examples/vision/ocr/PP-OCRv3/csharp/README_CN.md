[English](README.md) | 简体中文
# PPOCRv3 C#部署示例

本目录下提供`infer.cs`来调用C# API快速完成PPOCRv3模型在CPU/GPU上部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

在本目录执行如下命令即可在Windows完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

## 1. 下载C#包管理程序nuget客户端
> https://dist.nuget.org/win-x86-commandline/v6.4.0/nuget.exe

下载完成后将该程序添加到环境变量**PATH**中

## 2. 下载模型文件和测试图片
> https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar # (下载后解压缩)
> https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
> https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
> https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
> https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

## 3. 编译示例代码

本文档编译的示例代码可在解压的库中找到，编译工具依赖VS 2019的安装，**Windows打开x64 Native Tools Command Prompt for VS 2019命令工具**，通过如下命令开始编译

```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\ocr\PP-OCRv3\csharp

mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=D:\Download\fastdeploy-win-x64-gpu-x.x.x -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"

nuget restore
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

关于使用Visual Studio 2019创建sln工程，或者CMake工程等方式编译的更详细信息，可参考如下文档
- [在 Windows 使用 FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)
- [FastDeploy C++库在Windows上的多种使用方式](../../../../../docs/cn/faq/use_sdk_on_windows_build.md)

## 4. 运行可执行程序

注意Windows上运行时，需要将FastDeploy依赖的库拷贝至可执行程序所在目录, 或者配置环境变量。FastDeploy提供了工具帮助我们快速将所有依赖库拷贝至可执行程序所在目录,通过如下命令将所有依赖的dll文件拷贝至可执行程序所在的目录(可能生成的可执行文件在Release下还有一层目录，这里假设生成的可执行文件在Release处)
```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x

fastdeploy_init.bat install %cd% D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\ocr\PP-OCRv3\csharp\build\Release
```

将dll拷贝到当前路径后，准备好模型和图片，使用如下命令运行可执行程序即可
```shell
cd Release
# CPU推理
infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v3.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU推理
infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v3.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
```

## PPOCRv3 C#接口

### 模型

```c#
fastdeploy.vision.ocr.DBDetector(
        string model_file,
        string params_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> DBDetector模型加载和初始化。

> **参数**

>> * **model_file**(str): 模型文件路径
>> * **params_file**(str): 参数文件路径
>> * **runtime_option**(RuntimeOption): 后端推理配置，默认为null，即采用默认配置
>> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

```c#
fastdeploy.vision.ocr.Classifier(
        string model_file,
        string params_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> Classifier模型加载和初始化。

> **参数**

>> * **model_file**(str): 模型文件路径
>> * **params_file**(str): 参数文件路径
>> * **runtime_option**(RuntimeOption): 后端推理配置，默认为null，即采用默认配置
>> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

```c#
fastdeploy.vision.ocr.Recognizer(
        string model_file,
        string params_file,
        string label_path,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> Recognizer模型加载和初始化。

> **参数**

>> * **model_file**(str): 模型文件路径
>> * **params_file**(str): 参数文件路径
>> * **label_path**(str): 标签文件路径
>> * **runtime_option**(RuntimeOption): 后端推理配置，默认为null，即采用默认配置
>> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

```c#
fastdeploy.pipeline.PPOCRv3Model(
        DBDetector dbdetector,
        Classifier classifier,
        Recognizer recognizer)
```

> PPOCRv3Model模型加载和初始化。

> **参数**

>> * **det_model**(FD_C_DBDetectorWrapper*): DBDetector模型
>> * **cls_model**(FD_C_ClassifierWrapper*): Classifier模型
>> * **rec_model**(FD_C_RecognizerWrapper*): Recognizer模型文件

#### Predict函数

```c#
fastdeploy.OCRResult Predict(OpenCvSharp.Mat im)
```

> 模型预测接口，输入图像直接输出结果。
>
> **参数**
>
>> * **im**(Mat): 输入图像，注意需为HWC，BGR格式
>>
> **返回值**
>
>> * **result**: OCR预测结果,包括由检测模型输出的检测框位置,分类模型输出的方向分类,以及识别模型输出的识别结果, OCRResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
