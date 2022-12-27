[English](README_EN.md) | 简体中文
# PaddleClas C++部署示例

本目录下提供`infer.cc`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上ResNet50_vd推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

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
# GPU上TensorRT推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 2
# IPU推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 3
# KunlunXin XPU推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 4
# Huawei Ascend NPU推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 5
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- [如何使用华为昇腾NPU部署](../../../../../docs/cn/faq/use_sdk_on_ascend.md)

## PaddleClas C++接口

### PaddleClas类

```c++
fastdeploy::vision::classification::PaddleClasModel(
        const string& model_file,
        const string& params_file,
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PaddleClas模型加载和初始化，其中model_file, params_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

#### Predict函数

> ```c++
> PaddleClasModel::Predict(cv::Mat* im, ClassifyResult* result, int topk = 1)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 分类结果，包括label_id，以及相应的置信度, ClassifyResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)
> > * **topk**(int):返回预测概率最高的topk个分类结果，默认为1


- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
