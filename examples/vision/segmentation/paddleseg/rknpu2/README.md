# PaddleSeg 模型部署

## 模型版本说明

- [PaddleSeg develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

目前FastDeploy使用RKNPU2推理PPSeg支持如下模型的部署:

| 模型                                                                                                                                           | 参数文件大小 | 输入Shape  | mIoU   | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------------------------------------------------------------------------------------|:-------|:---------|:-------|:------------|:---------------|
| [Unet-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)                                       | 52MB   | 1024x512 | 65.00% | 66.02%      | 66.89%         |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)          | 31MB   | 1024x512 | 77.04% | 77.73%      | 77.46%         |
| [PP-HumanSegV1-Lite(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz)                                      | 543KB  | 192x192  | 86.2%  | -           | -              |
| [PP-HumanSegV2-Lite(通用人像分割模型)](https://bj.bcebos.com/paddle2onnx/libs/PP_HumanSegV2_Lite_192x192_infer.tgz)                                  | 12MB   | 192x192  | 92.52% | -           | -              |
| [PP-HumanSegV2-Mobile(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz)                          | 29MB   | 192x192  | 93.13% | -           | -              |
| [PP-HumanSegV1-Server(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz)                                  | 103MB  | 512x512  | 96.47% | -           | -              |
| [Portait-PP-HumanSegV2_Lite(肖像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz)               | 3.6M   | 256x144  | 96.63% | -           | -              |
| [FCN-HRNet-W18-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)                     | 37MB   | 1024x512 | 78.97% | 79.49%      | 79.74%         |
| [Deeplabv3-ResNet101-OS8-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) | 150MB  | 1024x512 | 79.90% | 80.22%      | 80.47%         |

## 准备PaddleSeg部署模型以及转换模型
RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:
* Paddle动态图模型转换为ONNX模型，请参考[PaddleSeg模型导出说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg)
* ONNX模型转换RKNN模型的过程，请参考[转换文档](../../../../../docs/cn/faq/rknpu2/export.md)进行转换。

## 模型转换example

* [PPHumanSeg](./pp_humanseg.md)

## 详细部署文档
- [RKNN总体部署教程](../../../../../docs/cn/faq/rknpu2/rknpu2.md)
- [C++部署](cpp)
- [Python部署](python)
