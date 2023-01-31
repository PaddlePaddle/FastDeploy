English | [简体中文](README_CN.md)

# FastestDet Ready-to-deploy Model

- The deployment of the FastestDet model is based on [FastestDet](https://github.com/dog-qiuqiu/FastestDet.git) and [Pre-trained Model Based on COCO 2017](https://github.com/dog-qiuqiu/FastestDet.git)
  - （1）The *.onnx provided by [Official Repository](https://github.com/dog-qiuqiu/FastestDet.git) can be deployed directly；
  - （2）The FastestDet   model trained by personal data should employ `test.py` in [FastestDet](https://github.com/dog-qiuqiu/FastestDet.git) to export the ONNX files for deployment.


## Download Pre-trained ONNX Model

For developers' testing, models exported by FastestDet are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy  | Note |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [FastestDet](https://bj.bcebos.com/paddlehub/fastdeploy/FastestDet.onnx) | 969KB | 25.3% | This model file is sourced from [FastestDet](https://github.com/dog-qiuqiu/FastestDet.git)，BSD-3-Clause license |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Release Note

- Document and code are based on [FastestDet](https://github.com/dog-qiuqiu/FastestDet.git) 