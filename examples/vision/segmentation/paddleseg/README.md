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
| [Unet_cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz) | 52MB | 1024x512 | 65.00% | 66.02% | 66.89% |
| [PPLCNetV2_base](https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNetV2_base_infer.tgz)  | 26MB  | 224x224 |77.04% | 93.27% |
| [EfficientNetB7](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB7_infer.tgz) |  255MB | 600x600 | 84.3% | 96.9% |
| [EfficientNetB0_small](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB0_small_infer.tgz)|  18MB | 224x224 | 75.8% | 75.8% |
| [GhostNet_x1_3_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x1_3_ssld_infer.tgz) |  29MB | 224x224 | 75.7% | 92.5% |
| [GhostNet_x0_5_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x0_5_infer.tgz) |  10MB | 224x224 | 66.8% | 86.9% |
| [MobileNetV1_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_x0_25_infer.tgz) |  1.9MB | 224x224 | 51.4% | 75.5% |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_ssld_infer.tgz) |  17MB | 224x224 | 77.9% | 93.9% |
| [MobileNetV2_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_x0_25_infer.tgz) |  5.9MB | 224x224 | 53.2% | 76.5% |
| [MobileNetV2_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_ssld_infer.tgz) |  14MB | 224x224 | 76.74% | 93.39% |
| [MobileNetV3_small_x0_35_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_small_x0_35_ssld_infer.tgz) |  6.4MB | 224x224 | 55.55% | 77.71% |
| [MobileNetV3_large_x1_0_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_large_x1_0_ssld_infer.tgz) |  22MB | 224x224 | 78.96% | 94.48% |
| [ShuffleNetV2_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x0_25_infer.tgz) |  2.4MB | 224x224 | 49.9% | 73.79% |
| [ShuffleNetV2_x2_0](https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x2_0_infer.tgz) |  29MB | 224x224 | 73.15% | 91.2% |
| [SqueezeNet1_1](https://bj.bcebos.com/paddlehub/fastdeploy/SqueezeNet1_1_infer.tgz) |  4.8MB | 224x224 | 60.1% | 81.9% |
| [InceptionV3](https://bj.bcebos.com/paddlehub/fastdeploy/InceptionV3_infer.tgz) |  92MB | 299x299 | 79.14% | 94.59% |
| [PPHGNet_tiny_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_tiny_ssld_infer.tgz) |  57MB | 224x224 | 81.95% | 96.12% |
| [PPHGNet_base_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_base_ssld_infer.tgz) |  274MB | 224x224 | 85.0% | 97.35% |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz) |  98MB | 224x224 | 79.12% | 94.44% |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
