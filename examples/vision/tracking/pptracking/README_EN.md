English | [简体中文](README.md)
# PP-Tracking Model Deployment

## Model Description

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

## List of Supported Models

Now FastDeploy supports the deployment of the following models

- [PP-Tracking models](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/mot)


## Export Deployment Models

Before deployment, the trained PP-Tracking needs to be exported into the deployment model. Refer to [Export Model](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/pptracking/cpp/README.md) for more details.


## Download Pre-trained Models

For developers' testing, PP-Tracking’s pedestrian tracking pendant model is provided below. Developers can download and use it directly. Refer to [PPTracking](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/pptracking/README_cn.md) for other models.

| Model                                                                                                   | Parameter Size   | Accuracy    | Note |
|:-----------------------------------------------------------------------------------------------------|:-------|:----- | :------ |
| [PP-Tracking](https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz) | 51.2MB | - |

**Statement**
- Only JDE models are supported（JDE，FairMOT，MCFairMOT)；
- 目前暂不支持SDE模型的部署，待PaddleDetection官方更新SED部署代码后，对SDE模型进行支持。


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
