English | [简体中文](README_CN.md)
# RobustVideoMatting Model Deployment

## Model Description

- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/commit/81a1093)

## List of Supported Models

Now FastDeploy supports the deployment of the following models

- [RobustVideoMatting model](https://github.com/PeterL1n/RobustVideoMatting)

## Download Pre-trained Models

For developers' testing, models exported by RobustVideoMatting are provided below. Developers can download and use them directly.

| Model                                                               | Parameter Size    | Accuracy    | Note |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [rvm_mobilenetv3_fp32.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_fp32.onnx) | 15MB ||exported from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/commit/81a1093)，GPL-3.0 License |
| [rvm_resnet50_fp32.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_resnet50_fp32.onnx) | 103MB | |exported from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/commit/81a1093)，GPL-3.0 License |
| [rvm_mobilenetv3_trt.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_trt.onnx) | 15MB | |exported from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/commit/81a1093)，GPL-3.0 License |
| [rvm_resnet50_trt.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_resnet50_trt.onnx) | 103MB | |exported from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/commit/81a1093)，GPL-3.0 License |

**Note**：
- If you want to use TensorRT for inference, download onnx model file with the trt suffix is necessary.

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
