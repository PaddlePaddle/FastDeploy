# PPMatting模型部署

## 模型版本说明

- [PPMatting Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PPMatting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)


## 导出部署模型

在部署前，需要先将PPMatting导出成部署模型，导出步骤参考文档[导出模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)

注意：在导出模型时不要进行NMS的去除操作，正常导出即可。

## 下载预训练模型

为了方便开发者的测试，下面提供了PPMatting导出的各系列模型，开发者可直接下载使用。

其中精度指标来源于PPMatting中对各模型的介绍，详情各参考PPMatting中的说明。


| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [PPMatting](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz) | 87MB | - |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
