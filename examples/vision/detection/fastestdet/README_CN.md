[English](README.md) | 简体中文
# FastestDet准备部署模型

- FastestDet部署模型实现来自[FastestDet](https://github.com/dog-qiuqiu/FastestDet.git),和[基于COCO的预训练模型](https://github.com/dog-qiuqiu/FastestDet.git)
  - （1）[官方库](https://github.com/dog-qiuqiu/FastestDet.git)提供的*.onnx可直接进行部署；
  - （2）开发者基于自己数据训练的 FastestDet模型，可使用[FastestDet](https://github.com/dog-qiuqiu/FastestDet.git)中的`test.py`导出ONNX文件后，完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了FastestDet导出的模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度  | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [FastestDet](https://bj.bcebos.com/paddlehub/fastdeploy/FastestDet.onnx) | 969KB | 25.3% | 此模型文件来源于[FastestDet](https://github.com/dog-qiuqiu/FastestDet.git)，BSD-3-Clause license |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[FastestDet](https://github.com/dog-qiuqiu/FastestDet.git) 编写
