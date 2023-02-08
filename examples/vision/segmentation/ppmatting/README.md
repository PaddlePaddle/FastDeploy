English | [简体中文](README_CN.md)
# PP-Matting Model Deployment

## Model Description

- [PP-Matting Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)

## List of Supported Models

Now FastDeploy supports the deployment of the following models

- [PP-Matting models](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
- [PP-HumanMatting models](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
- [ModNet models](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)


## Export Deployment Model

Before deployment, PP-Matting needs to be exported into the deployment model. Refer to [Export Model](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting) for more information. (Tips: You need to set the `--input_shape` parameter of the export script when exporting PP-Matting and PP-HumanMatting models)


## Download Pre-trained Models

For developers' testing, models exported by PP-Matting are provided below. Developers can download and use them directly.

The accuracy metric is sourced from the model description in PP-Matting. (Accuracy data are not provided) Refer to the introduction in PP-Matting for more details.

| Model                                                               | Parameter Size    | Accuracy    | Note |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [PP-Matting-512](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz) | 106MB | - |
| [PP-Matting-1024](https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-1024.tgz) | 106MB | - |
| [PP-HumanMatting](https://bj.bcebos.com/paddlehub/fastdeploy/PPHumanMatting.tgz) | 247MB | - |
| [Modnet-ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_ResNet50_vd.tgz) | 355MB | - |
| [Modnet-MobileNetV2](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_MobileNetV2.tgz) | 28MB | - |
| [Modnet-HRNet_w18](https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_HRNet_w18.tgz) | 51MB | - |



## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
