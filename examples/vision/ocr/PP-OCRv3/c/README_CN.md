[English](README.md) | 简体中文
# PPOCRv3 C部署示例

本目录下提供`infer.c`来调用C API快速完成PPOCRv3模型在CPU/GPU上部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

```
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j


# 下载模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- [如何使用华为昇腾NPU部署](../../../../../docs/cn/faq/use_sdk_on_ascend.md)

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">


## PPOCRv3 C API接口

### 配置

```c
FD_C_RuntimeOptionWrapper* FD_C_CreateRuntimeOptionWrapper()
```

> 创建一个RuntimeOption的配置对象，并且返回操作它的指针。
>
> **返回**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针


```c
void FD_C_RuntimeOptionWrapperUseCpu(
     FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper)
```

> 开启CPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针

```c
void FD_C_RuntimeOptionWrapperUseGpu(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int gpu_id)
```
> 开启GPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针
> * **gpu_id**(int): 显卡号


### 模型

```c

FD_C_DBDetectorWrapper* FD_C_CreateDBDetectorWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)

```

> 创建一个DBDetector的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
> * **fd_c_dbdetector_wrapper**(FD_C_DBDetectorWrapper*): 指向DBDetector模型对象的指针

FD_C_ClassifierWrapper* FD_C_CreateClassifierWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)

> 创建一个Classifier的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
>
> * **fd_c_classifier_wrapper**(FD_C_ClassifierWrapper*): 指向Classifier模型对象的指针

FD_C_RecognizerWrapper* FD_C_CreateRecognizerWrapper(
    const char* model_file, const char* params_file, const char* label_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)

> 创建一个Recognizer的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **label_path**(const char*): 标签文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
> * **fd_c_recognizer_wrapper**(FD_C_RecognizerWrapper*): 指向Recognizer模型对象的指针


FD_C_PPOCRv3Wrapper* FD_C_CreatePPOCRv3Wrapper(
    FD_C_DBDetectorWrapper* det_model,
    FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model
)

> 创建一个PPOCRv3的模型，并且返回操作它的指针。
>
> **参数**
>
> * **det_model**(FD_C_DBDetectorWrapper*): DBDetector模型
> * **cls_model**(FD_C_ClassifierWrapper*): Classifier模型
> * **rec_model**(FD_C_RecognizerWrapper*): Recognizer模型
>
> **返回**
>
> * **fd_c_ppocrv3_wrapper**(FD_C_PPOCRv3Wrapper*): 指向PPOCRv3模型对象的指针



#### 读写图像

```c
FD_C_Mat FD_C_Imread(const char* imgpath)
```

> 读取一个图像，并且返回cv::Mat的指针。
>
> **参数**
>
> * **imgpath**(const char*): 图像文件路径
>
> **返回**
>
> * **imgmat**(FD_C_Mat): 指向图像数据cv::Mat的指针。


```c
FD_C_Bool FD_C_Imwrite(const char* savepath,  FD_C_Mat img);
```

> 将图像写入文件中。
>
> **参数**
>
> * **savepath**(const char*): 保存图像的路径
> * **img**(FD_C_Mat): 指向图像数据的指针
>
> **返回**
>
> * **result**(FD_C_Bool): 表示操作是否成功


#### Predict函数

```c
FD_C_Bool FD_C_PPOCRv3WrapperPredict(
    FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper,
    FD_C_Mat img,
    FD_C_OCRResult* result)
```
>
> 模型预测接口，输入图像直接并生成分类结果。
>
> **参数**
> * **fd_c_ppocrv3_wrapper**(FD_C_PPOCRv3Wrapper*): 指向PPOCRv3模型的指针
> * **img**（FD_C_Mat）: 输入图像的指针，指向cv::Mat对象，可以调用FD_C_Imread读取图像获取
> * **result**FD_C_OCRResult*): OCR预测结果,包括由检测模型输出的检测框位置,分类模型输出的方向分类,以及识别模型输出的识别结果, OCRResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


#### Predict结果

```c
FD_C_OCRResultWrapper* FD_C_CreateOCRResultWrapperFromData(
    FD_C_OCRResult* fd_c_ocr_result)
```
>
> 创建一个FD_C_OCRResultWrapper对象的指针，FD_C_OCRResultWrapper中包含了C++的`fastdeploy::vision::OCRResult`对象，通过该指针，使用C API可以访问调用对应C++中的函数。
>
>
> **参数**
> * **fd_c_ocr_result**(FD_C_OCRResult*): 指向FD_C_OCRResult对象的指针
>
> **返回**
> * **fd_c_ocr_result_wrapper**(FD_C_OCRResultWrapper*): 指向FD_C_OCRResultWrapper的指针


```c
char* FD_C_OCRResultWrapperStr(
    FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper);
```
>
> 调用FD_C_OCRResultWrapper所包含的`fastdeploy::vision::OCRResult`对象的Str()方法，返回相关结果内数据信息的字符串。
>
> **参数**
> * **fd_c_ocr_result_wrapper**(FD_C_OCRResultWrapper*): 指向FD_C_OCRResultWrapper对象的指针
>
> **返回**
> * **str**(char*): 表示结果数据信息的字符串




## 其它文档

- [PPOCR 系列模型介绍](../../)
- [PPOCRv3 Python部署](../python)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
