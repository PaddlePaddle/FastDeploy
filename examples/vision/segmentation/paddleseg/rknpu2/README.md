English | [简体中文](README_CN.md)
# PaddleSeg Model Deployment

## Model Version

- [PaddleSeg develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

Currently FastDeploy using RKNPU2 to infer PPSeg supports the following model deployments:

| Model                                                                                                                                          | Parameter File Size | Input Shape  | mIoU   | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------------------------------------------------------------------------------------|:-------|:---------|:-------|:------------|:---------------|
| [Unet-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)                                       | 52MB   | 1024x512 | 65.00% | 66.02%      | 66.89%         |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)          | 31MB   | 1024x512 | 77.04% | 77.73%      | 77.46%         |
| [PP-HumanSegV1-Lite(Universal portrait segmentation model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz)                                      | 543KB  | 192x192  | 86.2%  | -           | -              |
| [PP-HumanSegV2-Lite(Universal portrait segmentation model)](https://bj.bcebos.com/paddle2onnx/libs/PP_HumanSegV2_Lite_192x192_infer.tgz)                                  | 12MB   | 192x192  | 92.52% | -           | -              |
| [PP-HumanSegV2-Mobile(Universal portrait segmentation model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz)                          | 29MB   | 192x192  | 93.13% | -           | -              |
| [PP-HumanSegV1-Server(Universal portrait segmentation model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz)                                  | 103MB  | 512x512  | 96.47% | -           | -              |
| [Portait-PP-HumanSegV2_Lite(Portrait segmentation model)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz)               | 3.6M   | 256x144  | 96.63% | -           | -              |
| [FCN-HRNet-W18-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)                     | 37MB   | 1024x512 | 78.97% | 79.49%      | 79.74%         |
| [Deeplabv3-ResNet101-OS8-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) | 150MB  | 1024x512 | 79.90% | 80.22%      | 80.47%         |

## Prepare PaddleSeg Deployment Model and Conversion Model
RKNPU needs to convert the Paddle model to RKNN model before deploying, the steps are as follows:
* For the conversion of Paddle dynamic diagram model to ONNX model, please refer to [PaddleSeg Model Export](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg).
* For the process of converting ONNX model to RKNN model, please refer to [Conversion document](../../../../../docs/en/faq/rknpu2/export.md).

## An example of Model Conversion

* [PPHumanSeg](./pp_humanseg_EN.md)

## Detailed Deployment Document
- [Overall RKNN Deployment Guidance](../../../../../docs/en/faq/rknpu2/rknpu2.md)
- [Deploy with C++](cpp)
- [Deploy with Python](python)
