[English](README.md) | 简体中文
# PP-Tracking模型部署

## 模型版本说明

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PP-Tracking系列模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/mot)


## 导出部署模型

在部署前，需要先将训练好的PP-Tracking导出成部署模型，导出PPTracking导出模型步骤，参考文档[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/pptracking/cpp/README.md)。


## 下载预训练模型

为了方便开发者的测试，下面提供了PP-Tracking行人跟踪垂类模型，开发者可直接下载使用，更多模型参见[PPTracking](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/pptracking/README_cn.md)。

| 模型                                                                                                   | 参数大小   | 精度    | 备注 |
|:-----------------------------------------------------------------------------------------------------|:-------|:----- | :------ |
| [PP-Tracking](https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz) | 51.2MB | - |

**说明**
- 仅支持JDE模型（JDE，FairMOT，MCFairMOT)；
- 目前暂不支持SDE模型的部署，待PaddleDetection官方更新SED部署代码后，对SDE模型进行支持。


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
