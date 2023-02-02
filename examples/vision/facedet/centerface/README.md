English | [简体中文](README_CN.md)

# CenterFace Ready-to-deploy Model

- The deployment of the CenterFace model is based on [CenterFace](https://github.com/Star-Clouds/CenterFace.git) and [Pre-trained Model Based on WIDER FACE](https://github.com/Star-Clouds/CenterFace.git)
  - （1）The *.onnx provided by [Official Repository](https://github.com/Star-Clouds/CenterFace.git) can be deployed directly；
  - （2）The CenterFace train code is not open source and users cannot train it.


## Download Pre-trained ONNX Model

For developers' testing, models exported by CenterFace are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository on WIDER FACE test set)
| Model                                                               | Size    | Accuracy(Easy Set,Medium Set,Hard Set)  | Note |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [CenterFace](https://bj.bcebos.com/paddlehub/fastdeploy/CenterFace.onnx) | 7.2MB | 93.2%,92.1%,87.3% | This model file is sourced from [CenterFace](https://github.com/Star-Clouds/CenterFace.git)，MIT license |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Release Note

- Document and code are based on [CenterFace](https://github.com/Star-Clouds/CenterFace.git) 