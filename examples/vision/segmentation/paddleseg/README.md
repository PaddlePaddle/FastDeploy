# PaddleSeg 模型部署

## 模型版本说明

- [PaddleSeg develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

目前FastDeploy支持如下模型的部署 

- [U-Net系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/unet/README.md)
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/pp_liteseg/README.md)
- [PP-HumanSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README.md)
- [FCN系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fcn/README.md)
- [DeepLabV3系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/deeplabv3/README.md)

【注意】如你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../matting)

## 准备PaddleSeg部署模型

PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_cn.md)  

**注意**
- PaddleSeg导出的模型包含`model.pdmodel`、`model.pdiparams`和`deploy.yaml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息
- aarch64平台（如：Jetson）暂时只支持`onnxruntime`和`tensorrt`作为后端推理（**不支持**非固定shape的图片输入即动态输入）。因此，**必须指定**`--input_shape`导出具有固定输入的PaddleSeg模型（FastDeploy会在预处理阶段，对原图进行resize操作）
- 在使用其他平台（如：Windows、Mac、Linux），在导出PaddleSeg模型模型时，可指定`--input_shape`参数（当想采用`onnxruntime`或`tensorrt`作为后端进行推理）。但是，若输入的预测图片尺寸并不固定，建议使用默认值即**不指定**该参数（同时采用Paddle Inference或者OpenVino作为后端进行推理）

## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型（导出方式为：**不指定**`--input_shape`，**指定**`--output_op none`），开发者可直接下载使用。

| 模型                                                               | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [Unet-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz) | 52MB | 1024x512 | 65.00% | 66.02% | 66.89% |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)  | 31MB  | 1024x512 |77.04% | 77.73% | 77.46% |
| [PP-HumanSegV1-Lite(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz) |  543KB | 192x192 | 86.2% | - | - |
| [PP-HumanSegV2-Lite(通用人像分割模型)](https://bj.bcebos.com/paddle2onnx/libs/PP_HumanSegV2_Lite_192x192_infer.tgz) |  12MB | 192x192 | 92.52% | - | - |
| [PP-HumanSegV2-Mobile(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz) |  29MB | 192x192 | 93.13% | - | - |
| [PP-HumanSegV1-Server(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz) |  103MB | 512x512 | 96.47% | - | - |
| [Portait-PP-HumanSegV2_Lite(肖像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz) |  3.6M | 256x144 | 96.63% | - | - |
| [FCN-HRNet-W18-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz) |  37MB | 1024x512 | 78.97% | 79.49% | 79.74% |
| [Deeplabv3-ResNet101-OS8-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) |  150MB | 1024x512 | 79.90% | 80.22% | 80.47% |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
