# EDVR模型部署

## 模型版本说明

- [PaddleDetection develop](https://github.com/PaddlePaddle/PaddleGAN)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [EDVR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)。


## 导出部署模型

在部署前，需要先将训练好的PP-MSVSR导出成部署模型，导出PP-MSVSR导出模型步骤，参考文档[导出模型](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)。


| 模型                                                                             | 参数大小   | 精度    | 备注 |
|:--------------------------------------------------------------------------------|:-------|:----- | :------ |
| [EDVR](https://bj.bcebos.com/paddlehub/fastdeploy/EDVR_M_wo_tsa_SRx4.tgz) | 14.9MB | - |

**注意**：非常不建议在没有独立显卡的设备上运行该模型

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
