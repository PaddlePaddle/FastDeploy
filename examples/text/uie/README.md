English | [简体中文](README_CN.md)

# Universal Information Extraction  UIE Model Deployment

## Introduction to the Model Version

- [PaddleNLP Universal Information Extraction（UIE）](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/uie)

## A List of Supported Models

| Model |  Structure  | Language |
| :---: | :--------: | :--------: |
| `uie-base`| 12-layers, 768-hidden, 12-heads | Chinese |
| `uie-medium`| 6-layers, 768-hidden, 12-heads | Chinese |
| `uie-mini`| 6-layers, 384-hidden, 12-heads | Chinese |
| `uie-micro`| 4-layers, 384-hidden, 12-heads | Chinese |
| `uie-nano`| 4-layers, 312-hidden, 12-heads | Chinese |


## Export Deployment Models

Before deployment, you need to export the UIE model into the deployment model. Please refer to [Export Model](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/uie#47-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2).

## Download Pre-trained Models

Models exported by UIE are provided below for developers' testing. Developers can directly download them.

The accuracy metric F1 is derived from the 0-shot experiments conducted in PaddleNLP on each model in the Internet self-built dataset. Please refer to instructions in PaddleNLP for more details.

| Model                                                               | Parameter Size    | F1 Value|
|:---------------------------------------------------------------- |:----- |:----- |
|[uie-base](https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz)| 416M | 78.33	|
|[uie-medium](https://bj.bcebos.com/fastdeploy/models/uie/uie-medium.tgz)| 265M | 78.32 |
|[uie-mini](https://bj.bcebos.com/fastdeploy/models/uie/uie-mini.tgz)| 95M | 72.09 |
|[uie-micro](https://bj.bcebos.com/fastdeploy/models/uie/uie-micro.tgz)| 83M | 66.00 |
|[uie-nano](https://bj.bcebos.com/fastdeploy/models/uie/uie-nano.tgz)| 64M | 62.86 |

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)
- [Serve Deployment](serving)
