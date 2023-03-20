English | [简体中文](README_CN.md)

# PaddleClas Model Deployment

## Model Description

- [PaddleClas Release/2.4](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4)

Now FastDeploy supports the deployment of the following models

- [PP-LCNet Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNet.md)
- [PP-LCNetV2 Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md)
- [EfficientNet Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)
- [GhostNet Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Mobile.md)
- [MobileNet Models(including v1,v2,v3)](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Mobile.md)
- [ShuffleNet Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Mobile.md)
- [SqueezeNet Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Others.md)
- [Inception Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Inception.md)
- [PP-HGNet Models](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-HGNet.md)
- [ResNet Models（including vd series）](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/ResNet_and_vd.md)

## Prepare PaddleClas Deployment Model

For PaddleClas model export, refer to [Model Export](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA).  

Attention：The model exported by PaddleClas contains two files, including `inference.pdmodel` and `inference.pdiparams`. However, it is necessary to prepare the generic [inference_cls.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/deploy/configs/inference_cls.yaml) file provided by PaddleClas to meet the requirements of deployment. FastDeploy will obtain from the yaml file the preprocessing information required during inference. FastDeploy will get the preprocessing information needed by the model from the yaml file. Developers can directly download this file. But they need to modify the configuration parameters in the yaml file based on personalized needs. Refer to the configuration information in the infer section of the PaddleClas model training [config.](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/ppcls/configs/ImageNet)


## Download Pre-trained Model

For developers' testing, some models exported by PaddleClas (including the inference_cls.yaml file) are provided below. Developers can download them directly.

| Model                                                               | Parameter File Size    |Input Shape |  Top1 | Top5 |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- |
| [PPLCNet_x1_0](https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNet_x1_0_infer.tgz) | 12MB | 224x224 |71.32% | 90.03% |
| [PPLCNetV2_base](https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNetV2_base_infer.tgz)  | 26MB  | 224x224 |77.04% | 93.27% |
| [EfficientNetB7](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB7_infer.tgz) |  255MB | 600x600 | 84.3% | 96.9% |
| [EfficientNetB0](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB0_infer.tgz)|  19MB | 224x224 | 77.38% | 93.31% |
| [EfficientNetB0_small](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB0_small_infer.tgz)|  18MB | 224x224 | 75.8% | 92.58% |
| [GhostNet_x1_3](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x1_3_infer.tgz) |  27MB | 224x224 | 75.79% | 92.54% |
| [GhostNet_x1_3_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x1_3_ssld_infer.tgz) |  29MB | 224x224 | 79.3% | 94.49% |
| [GhostNet_x0_5](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x0_5_infer.tgz) |  10MB | 224x224 | 66.8% | 86.9% |
| [MobileNetV1_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_x0_25_infer.tgz) |  1.9MB | 224x224 | 51.4% | 75.5% |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_ssld_infer.tgz) |  17MB | 224x224 | 77.9% | 93.9% |
| [MobileNetV2_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_x0_25_infer.tgz) |  5.9MB | 224x224 | 53.2% | 76.5% |
| [MobileNetV2](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_infer.tgz) |  13MB | 224x224 | 72.15% | 90.65% |
| [MobileNetV2_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_ssld_infer.tgz) |  14MB | 224x224 | 76.74% | 93.39% |
| [MobileNetV3_small_x1_0](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_small_x1_0_infer.tgz) |  11MB | 224x224 | 68.24% | 88.06% |
| [MobileNetV3_small_x0_35_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_small_x0_35_ssld_infer.tgz) |  6.4MB | 224x224 | 55.55% | 77.71% |
| [MobileNetV3_large_x1_0_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_large_x1_0_ssld_infer.tgz) |  22MB | 224x224 | 78.96% | 94.48% |
| [ShuffleNetV2_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x0_25_infer.tgz) |  2.4MB | 224x224 | 49.9% | 73.79% |
| [ShuffleNetV2_x2_0](https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x2_0_infer.tgz) |  29MB | 224x224 | 73.15% | 91.2% |
| [SqueezeNet1_1](https://bj.bcebos.com/paddlehub/fastdeploy/SqueezeNet1_1_infer.tgz) |  4.8MB | 224x224 | 60.1% | 81.9% |
| [InceptionV3](https://bj.bcebos.com/paddlehub/fastdeploy/InceptionV3_infer.tgz) |  92MB | 299x299 | 79.14% | 94.59% |
| [PPHGNet_tiny_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_tiny_ssld_infer.tgz) |  57MB | 224x224 | 81.95% | 96.12% |
| [PPHGNet_small](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_small_infer.tgz) |  87MB | 224x224 | 81.51% | 95.82% |
| [PPHGNet_base_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_base_ssld_infer.tgz) |  274MB | 224x224 | 85.0% | 97.35% |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz) |  98MB | 224x224 | 79.12% | 94.44% |
| [ResNet50](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_infer.tgz) |  91MB | 224x224 | 76.5% | 93% |
| [ResNeXt50_32x4d](https://bj.bcebos.com/paddlehub/fastdeploy/ResNeXt50_32x4d_infer.tgz) |  89MB | 224x224 | 77.75% | 93.82% |
| [DenseNet121](https://bj.bcebos.com/paddlehub/fastdeploy/DenseNet121_infer.tgz) |  29MB | 224x224 | 75.66% | 92.58% |
| [PULC_person_exists](https://bj.bcebos.com/paddlehub/fastdeploy/person_exists_infer.tgz) |  6MB | 224x224 |  |  |
| [ViT_large_patch16_224](https://bj.bcebos.com/paddlehub/fastdeploy/ViT_large_patch16_224_infer.tgz) |  1.1GB | 224x224 | 83.23% |  96.50%|

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)
- [Serving Deployment](serving)
