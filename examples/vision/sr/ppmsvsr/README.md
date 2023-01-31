English | [简体中文](README_CN.md)
# PP-MSVSR Model Deployment

## Model Description

- [PaddleGAN develop](https://github.com/PaddlePaddle/PaddleGAN)

## List of Supported Models

Now FastDeploy supports the deployment of the following models

- [PP-MSVSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)


##  Export Deployment Model

Before deployment, export the trained PP-MSVSR to the deployment model. Refer to [Export Model](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md) for detailed steps.


| Model                                                                          | Parameter Size  | Accuracy    | Note |
|:----------------------------------------------------------------------------|:------|:----- | :------ |
| [PP-MSVSR](https://bj.bcebos.com/paddlehub/fastdeploy/PP-MSVSR_reds_x4.tar) | 8.8MB | - |


## Detailed Deployment Tutorials

- [Python deployment](python)
- [C++ deployment](cpp)
