[English](README.md) | 简体中文
# PaddleClas C 部署示例

本目录下提供`infer_xxx.c`来调用C API快速完成PaddleClas系列模型在CPU/GPU上部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上ResNet50_vd推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0
# GPU推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- [如何使用华为昇腾NPU部署](../../../../../docs/cn/faq/use_sdk_on_ascend.md)

## PaddleClas C API接口

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

FD_C_PaddleClasModelWrapper* FD_C_CreatePaddleClasModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* runtime_option,
    const FD_C_ModelFormat model_format)

```

> 创建一个PaddleClas的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **config_file**(const char*): 配置文件路径，即PaddleClas导出的部署yaml文件
> * **runtime_option**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
> * **fd_c_ppclas_wrapper**(FD_C_PaddleClasModelWrapper*): 指向PaddleClas模型对象的指针


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
FD_C_Bool FD_C_PaddleClasModelWrapperPredict(
    __fd_take FD_C_PaddleClasModelWrapper* fd_c_ppclas_wrapper, FD_C_Mat img,
    FD_C_ClassifyResult* fd_c_ppclas_result)
```
>
> 模型预测接口，输入图像直接并生成分类结果。
>
> **参数**
> * **fd_c_ppclas_wrapper**(FD_C_PaddleClasModelWrapper*): 指向PaddleClas模型的指针
> * **img**（FD_C_Mat）: 输入图像的指针，指向cv::Mat对象，可以调用FD_C_Imread读取图像获取
> * **fd_c_ppclas_result**（FD_C_ClassifyResult*): 分类结果，包括label_id，以及相应的置信度, ClassifyResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


#### Predict结果

```c
FD_C_ClassifyResultWrapper* FD_C_CreateClassifyResultWrapperFromData(
    FD_C_ClassifyResult* fd_c_classify_result)
```
>
> 创建一个FD_C_ClassifyResultWrapper对象的指针，FD_C_ClassifyResultWrapper中包含了C++的`fastdeploy::vision::ClassifyResult`对象，通过该指针，使用C API可以访问调用对应C++中的函数。
>
>
> **参数**
> * **fd_c_classify_result**(FD_C_ClassifyResult*): 指向FD_C_ClassifyResult对象的指针
>
> **返回**
> * **fd_c_classify_result_wrapper**(FD_C_ClassifyResultWrapper*): 指向FD_C_ClassifyResultWrapper的指针


```c
char* FD_C_ClassifyResultWrapperStr(
    FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper);
```
>
> 调用FD_C_ClassifyResultWrapper所包含的`fastdeploy::vision::ClassifyResult`对象的Str()方法，返回相关结果内数据信息的字符串。
>
> **参数**
> * **fd_c_classify_result_wrapper**(FD_C_ClassifyResultWrapper*): 指向FD_C_ClassifyResultWrapper对象的指针
>
> **返回**
> * **str**(char*): 表示结果数据信息的字符串




- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
