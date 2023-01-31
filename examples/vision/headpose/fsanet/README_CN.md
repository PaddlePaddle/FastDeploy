[English](README.md) | 简体中文
# FSANet 模型部署

## 模型版本说明

- [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch/commit/002549c)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [FSANet 模型](https://github.com/omasaht/headpose-fsanet-pytorch)

## 下载预训练模型

为了方便开发者的测试，下面提供了PFLD导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [fsanet-1x1.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/fsanet-1x1.onnx) | 1.2M | - |
| [fsanet-var.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/fsanet-var.onnx) | 1.2MB | - |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
