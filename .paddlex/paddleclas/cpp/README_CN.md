[English](README.md) | 简体中文
# PaddleClas C++部署示例

本目录下提供`infer.cc`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

```bash
# 找到部署包内的模型路径，例如ResNet50

# 准备一张测试图片，例如test.jpg

# CPU推理
./infer_demo ResNet50 test.jpg 0
# GPU推理
./infer_demo ResNet50 test.jpg 1
# GPU上TensorRT推理
./infer_demo ResNet50 test.jpg 2
# IPU推理
./infer_demo ResNet50 test.jpg 3
# KunlunXin XPU推理
./infer_demo ResNet50 test.jpg 4
# Huawei Ascend NPU推理
./infer_demo ResNet50 test.jpg 5
```

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
> > * **result**: 分类结果，包括label_id，以及相应的置信度, ClassifyResult说明参考[视觉模型预测结果](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/classification_result_CN.md)
> > * **topk**(int):返回预测概率最高的topk个分类结果，默认为1
