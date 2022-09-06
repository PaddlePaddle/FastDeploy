# PaddleSeg 模型部署

## 模型版本说明

- [PaddleSeg Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6)

目前FastDeploy支持如下模型的部署 

- [U-Net系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/unet/README.md)
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/pp_liteseg/README.md)
- [PP-HumanSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README.md)
- [FCN系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fcn/README.md)
- [DeepLabV3系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/deeplabv3/README.md)

## 准备PaddleSeg部署模型

PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_cn.md)  

注意：在使用PaddleSeg模型导出时，可指定`--input_shape`参数，若预测输入图片尺寸并不固定，建议使用默认值即不指定该参数。PaddleSeg导出的模型包含`model.pdmodel`、`model.pdiparams`和`deploy.yaml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息。


## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型（导出方式为：**不指定**`input_shape`和`with_softmax`，**指定**`without_argmax`），开发者可直接下载使用。

| 模型                                                               | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [Unet-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz) | 52MB | 1024x512 | 65.00% | 66.02% | 66.89% |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)  | 31MB  | 1024x512 |73.10% | 73.89% | - |
| [PP-HumanSegV1-Lite](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz) |  543KB | 192x192 | 86.2% | - | - |
| [PP-HumanSegV1-Server](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz) |  103MB | 512x512 | 96.47% | - | - |
| [FCN-HRNet-W18-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz) |  37MB | 1024x512 | 78.97% | 79.49% | 79.74% |
| [Deeplabv3-ResNet50-OS8-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet50_OS8_cityscapes_without_argmax_infer.tgz) |  150MB | 1024x512 | 79.90% | 80.22% | 80.47% |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
