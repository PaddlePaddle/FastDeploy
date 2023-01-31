English | [简体中文](README_CN.md)
# EDVR Model Deployment

## Model Description

- [PaddleGAN develop](https://github.com/PaddlePaddle/PaddleGAN)

## List of Supported Models

Now FastDeploy supports the deployment of the following models

- [EDVR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)。


## Export Deployment Model

Before deployment, export the trained EDVR to the deployment model. Refer to [Export Model](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md) for detailed steps.


| Model                                                                             | Parameter Size   | Accuracy    | Note |
|:--------------------------------------------------------------------------------|:-------|:----- | :------ |
| [EDVR](https://bj.bcebos.com/paddlehub/fastdeploy/EDVR_M_wo_tsa_SRx4.tar) | 14.9MB | - |

**Attention**: Running this model on a device without separate graphics card is highly discouraged

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
