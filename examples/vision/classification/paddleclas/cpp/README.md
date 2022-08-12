# PaddleClas C++部署示例

本目录下提供`infer.cc`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/quick_start/requirements.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/compile/prebuilt_libraries.md)

以Linux上ResNet50_vd推理为例，在本目录执行如下命令即可完成编译测试

```
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/paddlehub/fastdeploy/libs/0.2.0/fastdeploy-linux-x64-gpu-0.2.0.tgz
tar xvf fastdeploy-linux-x64-gpu-0.2.0.tgz
cd fastdeploy-linux-x64-gpu-0.2.0/examples/vision/classification/paddleclas/cpp
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../fastdeploy-linux-x64-gpu-0.2.0 
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
```

## PaddleClas C++接口

### PaddleClas类

```
fastdeploy::vision::classification::PaddleClasModel(
        const string& model_file,
        const string& params_file,
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE)
```

PaddleClas模型加载和初始化，其中model_file, params_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为Paddle格式

#### Predict函数

> ```
> PaddleClasModel::Predict(cv::Mat* im, ClassifyResult* result, int topk = 1)
> ```
> 
> 模型预测接口，输入图像直接输出检测结果。
> 
> **参数**
> 
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 分类结果，包括label_id，以及相应的置信度, ClassifyResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)
> > * **topk**(int):返回预测概率最高的topk个分类结果


- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
