# AdaFace准备部署模型

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/)
  - （1）[官方库](https://github.com/PaddlePaddle/PaddleClas/)中训练过后的Paddle模型[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的PaddleClas人脸识别模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 简介
一直以来，低质量图像的人脸识别都具有挑战性，因为低质量图像的人脸属性是模糊和退化的。将这样的图片输入模型时，将不能很好的实现分类。
而在人脸识别任务中，我们经常会利用opencv的仿射变换来矫正人脸数据，这时数据会出现低质量退化的现象。如何解决低质量图片的分类问题成为了模型落地时的痛点问题。

在AdaFace这项工作中，作者在损失函数中引入了另一个因素，即图像质量。作者认为，强调错误分类样本的策略应根据其图像质量进行调整。
具体来说，简单或困难样本的相对重要性应该基于样本的图像质量来给定。据此作者提出了一种新的损失函数来通过图像质量强调不同的困难样本的重要性。

由上，AdaFace缓解了低质量图片在输入网络后输出结果精度变低的情况，更加适合在人脸识别任务落地中使用。


## 导出ONNX模型
以AdaFace为例:
训练和导出代码，请参考[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/4479879?contributionType=1)


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了我转换过的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）其中精度指标来源于AIStudio中对各模型的介绍。

| 模型                                                                                                          | 大小    | 精度 (AgeDB_30) |
|:------------------------------------------------------------------------------------------------------------|:------|:--------------|
| [AdaFace-MobileFacenet](https://bj.bcebos.com/fastdeploy/models/onnx/mobile_face_net_ada_face_112x112.onnx) | 3.2MB | 95.5          |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
