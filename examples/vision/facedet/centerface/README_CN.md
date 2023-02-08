[English](README.md) | 简体中文
# CenterFace准备部署模型

- CenterFace部署模型实现来自[CenterFace](https://github.com/Star-Clouds/CenterFace.git),和[基于WIDER FACE的预训练模型](https://github.com/Star-Clouds/CenterFace.git)
  - （1）[官方库](https://github.com/Star-Clouds/CenterFace.git)提供的*.onnx可直接进行部署；
  - （2）由于CenterFace未开放训练源代码，开发者无法基于自己的数据训练CenterFace模型


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了CenterFace导出的模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库在WIDER FACE测试集上的结果）
| 模型                                                               | 大小    | 精度(Easy Set,Medium Set,Hard Set)  | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [CenterFace](https://bj.bcebos.com/paddlehub/fastdeploy/CenterFace.onnx) | 7.2MB | 93.2%,92.1%,87.3% | 此模型文件来源于[CenterFace](https://github.com/Star-Clouds/CenterFace.git)，MIT license |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[CenterFace](https://github.com/Star-Clouds/CenterFace.git) 编写