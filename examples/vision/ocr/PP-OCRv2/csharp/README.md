English | [简体中文](README_CN.md)
# PPOCRv2 C# Deployment Example

This directory provides `infer.cs` to finish the deployment of PPOCRv2 on CPU/GPU.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Please follow below instructions to compile and test in Windows. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4) is required to support this model.

## 1. Download C# package management tool nuget client
> https://dist.nuget.org/win-x86-commandline/v6.4.0/nuget.exe

Add nuget program into system variable **PATH**

## 2. Download model and image for test
> https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar (Decompress it)
> https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
> https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
> https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
> https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

## 3. Compile example code

Open `x64 Native Tools Command Prompt for VS 2019` command tool on Windows, cd to the demo path of ppyoloe and execute commands

```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\ocr\PP-OCRv2\csharp

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

fastdeploy_init.bat install %cd% D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\ocr\PP-OCRv2\csharp\build\Release
```

Then you can run your program and test the model with image
```shell
cd Release
# CPU inference
infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU inference
infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
```

## PPOCRv2 C# Interface

### Model Class

```c#
fastdeploy.vision.ocr.DBDetector(
        string model_file,
        string params_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> DBDetector initialization

> **Params**

>> * **model_file**(str):  Model file path
>> * **params_file**(str): Parameter file path
>> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
>> * **model_format**(ModelFormat): Model format.

```c#
fastdeploy.vision.ocr.Classifier(
        string model_file,
        string params_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> Classifier initialization

> **Params**

>> * **model_file**(str):  Model file path
>> * **params_file**(str): Parameter file path
>> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
>> * **model_format**(ModelFormat): Model format.

```c#
fastdeploy.vision.ocr.Recognizer(
        string model_file,
        string params_file,
        string label_path,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> Recognizer initialization

> **Params**

>> * **model_file**(str):  Model file path
>> * **params_file**(str): Parameter file path
>> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
>> * **model_format**(ModelFormat): Model format.

```c#
fastdeploy.pipeline.PPOCRv2Model(
        DBDetector dbdetector,
        Classifier classifier,
        Recognizer recognizer)
```

> PPOCRv2Model initialization

> **Params**

>> * **det_model**(FD_C_DBDetectorWrapper*): DBDetector model
>> * **cls_model**(FD_C_ClassifierWrapper*): Classifier model
>> * **rec_model**(FD_C_RecognizerWrapper*): Recognizer model

#### Predict Function

```c#
fastdeploy.OCRResult Predict(OpenCvSharp.Mat im)
```

> Model prediction interface. Input images and output results directly.
>
> **Params**
>
>> * **im**(Mat): Input images in HWC or BGR format
>>
> **Return**
>
>> * **result**: OCR prediction results, including the position of the detection box from the detection model, the classification of the direction from the classification model, and the recognition result from the recognition model. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for OCRResult

## Other Documents

- [PPOCR Model Description](../../)
- [PPOCRv2 Python Deployment](../python)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
