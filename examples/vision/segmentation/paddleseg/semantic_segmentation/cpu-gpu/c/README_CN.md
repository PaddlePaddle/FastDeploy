[English](README.md) | 简体中文
# PaddleSeg CPU-GPU C部署示例

本目录下提供`infer.c`来调用C API快速完成PaddleSeg模型在CPU/GPU上部署的示例。

## 1. 说明  
PaddleSeg支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署Segmentation模型。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleSeg部署模型](../README.md)，如果你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../matting)。

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PP-LiteSeg模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png


# CPU推理
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 0
# GPU推理
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 1
```

运行完成可视化结果如下图所示

<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)

如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- [如何使用华为昇腾NPU部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_ascend.md)

## 5. PaddleSeg C API接口

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
FD_C_PaddleSegWrapper* FD_C_CreatePaddleSegWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```
> 创建一个PaddleSeg的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **config_file**(const char*): 配置文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
>
> * **fd_c_ppseg_wrapper**(FD_C_PaddleSegWrapper*): 指向PaddleSeg模型对象的指针



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
FD_C_Bool FD_C_PaddleSegWrapperPredict(
    FD_C_PaddleSegWrapper* fd_c_ppseg_wrapper,
    FD_C_Mat img,
    FD_C_SegmentationResult* result)
```
>
> 模型预测接口，输入图像直接并生成分类结果。
>
> **参数**
> * **fd_c_ppseg_wrapper**(FD_C_PaddleSegWrapper*): 指向PaddleSeg模型的指针
> * **img**（FD_C_Mat）: 输入图像的指针，指向cv::Mat对象，可以调用FD_C_Imread读取图像获取
> * **result**FD_C_SegmentationResult*): Segmentation检测结果，SegmentationResult说明参考[视觉模型预测结果](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/)


#### Predict结果

```c
FD_C_Mat FD_C_VisSegmentation(FD_C_Mat im,
                              FD_C_SegmentationResult* result,
                              float weight)
```
>
> 对结果进行可视化，返回可视化的图像。
>
> **参数**
> * **im**(FD_C_Mat): 指向输入图像的指针
> * **segmentation_result**(FD_C_SegmentationResult*): 指向 FD_C_SegmentationResult结构的指针
> * **weight**(float): 透明度权重
>
> **返回**
> * **vis_im**(FD_C_Mat): 指向可视化图像的指针


## 6. 常见问题

- [PPSegmentation 系列模型介绍](../../)
- [PaddleSeg Python部署](../python)
- [模型预测结果说明](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/)
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
