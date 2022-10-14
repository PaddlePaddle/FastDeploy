# PP-Matting模型部署

## 模型版本说明

- [PP-Matting Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PP-Matting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
- [PP-HumanMatting系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
- [ModNet系列模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)


## 导出部署模型

在部署前，需要先将PP-Matting导出成部署模型，导出步骤参考文档[导出模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)(Tips:导出PP-Matting系列模型和PP-HumanMatting系列模型需要设置导出脚本的`--input_shape`参数)


## 下载预训练模型

为了方便开发者的测试，下面提供了PP-Matting导出的各系列模型，开发者可直接下载使用。

其中精度指标来源于PP-Matting中对各模型的介绍(未提供精度数据)，详情各参考PP-Matting中的说明。


| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [PP-Matting-512](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz) | 106MB | - |
| [PP-Matting-1024](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-1024.tgz) | 106MB | - |
| [PP-HumanMatting](https://bj.bcebos.com/paddlehub/fastdeploy/PPHumanMatting.tgz) | 247MB | - |
| [Modnet-ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_ResNet50_vd.tgz) | 355MB | - |
| [Modnet-MobileNetV2](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_MobileNetV2.tgz) | 28MB | - |
| [Modnet-HRNet_w18](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_HRNet_w18.tgz) | 51MB | - |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
