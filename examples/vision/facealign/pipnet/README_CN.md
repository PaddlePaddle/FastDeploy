[English](README.md) | 简体中文
# PIPNet 模型部署

## 模型版本说明

- [PIPNet](https://github.com/jhb86253817/PIPNet/tree/b9eab58)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PIPNet 模型](https://github.com/jhb86253817/PIPNet)

## 下载预训练模型

为了方便开发者的测试，下面提供了PIPNet导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [PIPNet19_ResNet18_AFLW](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet18_10x19x32x256_aflw.onnx) | 45.6M | - |
| [PIPNet29_ResNet18_COFW](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet18_10x29x32x256_cofw.onnx) | 46.1M | - |
| [PIPNet68_ResNet18_300W](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet18_10x68x32x256_300w.onnx) | 47.9M | - |
| [PIPNet98_ResNet18_WFLW](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet18_10x98x32x256_wflw.onnx) | 49.3M | - |
| [PIPNet19_ResNet101_AFLW](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet101_10x19x32x256_aflw.onnx) | 173.4M | - |
| [PIPNet29_ResNet101_COFW](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet101_10x29x32x256_cofw.onnx) | 175.3M | - |
| [PIPNet68_ResNet101_300W](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet101_10x68x32x256_300w.onnx) | 182.6M | - |
| [PIPNet98_ResNet101_WFLW](https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet101_10x98x32x256_wflw.onnx) | 188.3M | - |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
