# PaddleClas 模型部署

## 模型版本说明

- [PaddleClas Release/2.4](https://github.com/PaddlePaddle/PaddleClas)

## 准备PaddleClas部署模型

PaddleClas模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)  

注意：PaddleClas导出的模型仅包含`inference.pdmodel`和`inference.pdiparams`两个文档，但为了满足部署的需求，同时也需准备其提供的[inference_cls.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/deploy/configs/inference_cls.yaml)文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息，开发者可直接下载此文件使用。但需根据自己的需求修改yaml文件中的配置参数。


## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleClas导出的部分模型（含inference_cls.yaml文件），开发者可直接下载使用。

| 模型                                                               | 大小    |输入Shape |  精度    |
|:---------------------------------------------------------------- |:----- |:----- | :----- |
| [PPLCNet]() | 141MB | 224x224 |51.4% |
| [PPLCNetv2]()  | 10MB  | 224x224 |51.4% |
| [EfficientNet]() |     | 224x224 |     |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
