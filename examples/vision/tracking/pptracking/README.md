English | [简体中文](README_CN.md)
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
- SDE model deployment is not supported at present. Its deployment can be allowed after PaddleDetection officially updates SED deployment code.


## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
