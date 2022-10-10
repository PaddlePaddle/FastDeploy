# PPMatting模型部署

## 模型版本说明

- [PPMatting Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PPMatting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
- [PPHumanMatting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
- [ModNet系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)


## 导出部署模型

在部署前，需要先将PPMatting导出成部署模型，导出步骤参考文档[导出模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)(Tips:导出PPMatting系列模型和PPHumanMatting系列模型需要设置导出脚本的`--input_shape`参数)


## 下载预训练模型

为了方便开发者的测试，下面提供了PPMatting导出的各系列模型，开发者可直接下载使用。

其中精度指标来源于PPMatting中对各模型的介绍(未提供精度数据)，详情各参考PPMatting中的说明。


| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [PPMatting-512](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz) | 106MB | - |
| [PPMatting-1024](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-1024.tgz) | 106MB | - |
| [PPHumanMatting](https://bj.bcebos.com/paddlehub/fastdeploy/PPHumanMatting.tgz) | 247MB | - |
| [Modnet_ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_ResNet50_vd.tgz) | 355MB | - |
| [Modnet_MobileNetV2](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_MobileNetV2.tgz) | 28MB | - |
| [Modnet_HRNet_w18](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_HRNet_w18.tgz) | 51MB | - |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
