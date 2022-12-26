English | [简体中文](README.md)
# PaddleSeg Model Deployment

## Model Description

- [PaddleSeg develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

FastDeploy currently supports the deployment of the following models

- [U-Net models](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/unet/README.md)
- [PP-LiteSeg models](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/pp_liteseg/README.md)
- [PP-HumanSeg models](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README.md)
- [FCN models](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fcn/README.md)
- [DeepLabV3 models](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/deeplabv3/README.md)

【Attention】For **PP-Matting**、**PP-HumanMatting** and **ModNet** deployment, please refer to [Matting Model Deployment](../../matting)

## Prepare PaddleSeg Deployment Model

For the export of the PaddleSeg model, refer to [Model Export](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md) for more information

**Attention**
- The exported PaddleSeg model contains three files, including `model.pdmodel`、`model.pdiparams` and `deploy.yaml`. FastDeploy will get the pre-processing information for inference from yaml files.

## Download Pre-trained Model

For developers' testing, part of the PaddleSeg exported models are provided below. 
- without-argmax export mode: **Not specified**`--input_shape`，**specified**`--output_op none`
- with-argmax export mode：**Not specified**`--input_shape`，**specified**`--output_op argmax`

Developers can download directly. 


| Model                                                               | Parameter Size    | Input Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [Unet-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_with_argmax_infer.tgz) \| [Unet-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)  | 52MB | 1024x512 | 65.00% | 66.02% | 66.89% |
| [PP-LiteSeg-B(STDC2)-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz) \| [PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz) | 31MB  | 1024x512 | 79.04% |	79.52% | 79.85% |
|[PP-HumanSegV1-Lite-with-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV1_Lite_with_argmax_infer.tgz) \| [PP-HumanSegV1-Lite-without-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz) |  543KB | 192x192 | 86.2% | - | - |
|[PP-HumanSegV2-Lite-with-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Lite_192x192_with_argmax_infer.tgz) \| [PP-HumanSegV2-Lite-without-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Lite_192x192_infer.tgz) |  12MB | 192x192 | 92.52% | - | - |
| [PP-HumanSegV2-Mobile-with-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_with_argmax_infer.tgz) \| [PP-HumanSegV2-Mobile-without-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz) |  29MB | 192x192 | 93.13% | - | - |
|[PP-HumanSegV1-Server-with-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_with_argmax_infer.tgz) \| [PP-HumanSegV1-Server-without-argmax(General Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz) |  103MB | 512x512 | 96.47% | - | - |
| [Portait-PP-HumanSegV2-Lite-with-argmax(Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer.tgz) \| [Portait-PP-HumanSegV2-Lite-without-argmax(Portrait Segmentation Model)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz) |  3.6M | 256x144 | 96.63% | - | - |
| [FCN-HRNet-W18-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_with_argmax_infer.tgz) \| [FCN-HRNet-W18-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)(GPU inference for ONNXRuntime is not supported now) |  37MB | 1024x512 | 78.97% | 79.49% | 79.74% |
| [Deeplabv3-ResNet101-OS8-cityscapes-with-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer.tgz) \| [Deeplabv3-ResNet101-OS8-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) |  150MB | 1024x512 | 79.90% | 80.22% | 80.47% |

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
