[English](README.md) | 简体中文
# PP-MSVSR模型部署

## 模型版本说明

- [PaddleGAN develop](https://github.com/PaddlePaddle/PaddleGAN)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PP-MSVSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)。


## 导出部署模型

在部署前，需要先将训练好的PP-MSVSR导出成部署模型，导出PP-MSVSR导出模型步骤，参考文档[导出模型](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)。


| 模型                                                                          | 参数大小  | 精度    | 备注 |
|:----------------------------------------------------------------------------|:------|:----- | :------ |
| [PP-MSVSR](https://bj.bcebos.com/paddlehub/fastdeploy/PP-MSVSR_reds_x4.tar) | 8.8MB | - |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
