English | [简体中文](README_CN.md)
# AdaFace Ready-to-deploy Model

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/)
  - Paddle models trained in the [official repository](https://github.com/PaddlePaddle/PaddleClas/) are available for deployment after exporting Paddle static graph models；

## Introduction
Face recognition of low-quality images has been challenging because the face attributes of these images are blurred and degraded. We cannot realize optimal classification after such images are fed into the model. 
In face recognition, we often use the affine transformation of opencv to correct the face data that will degrade in low quality. The classification of low-quality images becomes a crucial problem during model development.

In AdaFace, we introduce another factor in the loss function, namely image quality. We maintain that the strategy of emphasizing misclassified samples should be adjusted according to their image quality. 
Specifically, the importance of easy and difficult samples should be given based on the their image quality. Accordingly, we propose a new loss function to emphasize the importance of different difficult samples by their image quality.

In a nutshell, AdaFace improves the situation that low-quality images become less accurate after input to the network, which is more effective in the task of face recognition.

## Export Paddle Static Graph Model
Taking AdaFace as an example:
Refer to [AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/4479879?contributionType=1) for training and export of the code.


## Download Pre-trained Paddle Static Graph Model

For developers' testing, converted models are provided below. Developers can download and use them directly. (The accuracy of the models in the table is sourced from the model introduction in AIStudio)

| Model                                                                                            | Size    | Accuracy (AgeDB_30) |
|:----------------------------------------------------------------------------------------------|:------|:--------------|
| [AdaFace-MobileFacenet](https://bj.bcebos.com/paddlehub/fastdeploy/mobilefacenet_adaface.tgz) | 3.2MB | 95.5          |

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
