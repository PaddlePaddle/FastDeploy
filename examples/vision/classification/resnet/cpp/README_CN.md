[English](README.md) | 简体中文
# ResNet C++部署示例

本目录下提供`infer.cc`快速完成ResNet系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上 ResNet50 推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载ResNet模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/resnet50.onnx
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU推理
./infer_demo resnet50.onnx ILSVRC2012_val_00000010.jpeg 0
# GPU推理
./infer_demo resnet50.onnx ILSVRC2012_val_00000010.jpeg 1
# GPU上TensorRT推理
./infer_demo resnet50.onnx ILSVRC2012_val_00000010.jpeg 2
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## ResNet C++接口

### ResNet类

```c++

fastdeploy::vision::classification::ResNet(
        const std::string& model_file,
        const std::string& params_file = "",
        const RuntimeOption& custom_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```


**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> ResNet::Predict(cv::Mat* im, ClassifyResult* result, int topk = 1)
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
