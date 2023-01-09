English | [简体中文](README_CN.md)
# NanoDetPlus Ready-to-deploy Model


- NanoDetPlus deployment is based on the code of [NanoDetPlus](https://github.com/RangiLyu/nanodet/tree/v1.0.0-alpha-1) and coco's [Pre-trained Model](https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1).

  - （1）The *.onnx provided by [official repository](https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1) can directly conduct the deployment；
  - （2）Models trained by developers should export ONNX models. Please refer to [Detailed Deployment Documents](#Detailed-Deployment-Documents) for deployment.

## Download Pre-trained ONNX Model

For developers' testing, models exported by NanoDetPlus are provided below. Developers can download them directly. (The model accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [NanoDetPlus_320](https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx ) | 4.6MB | 27.0% |
| [NanoDetPlus_320_sim](https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320-sim.onnx) | 4.6MB | 27.0% |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- Document and code are based on [NanoDetPlus v1.0.0-alpha-1](https://github.com/RangiLyu/nanodet/tree/v1.0.0-alpha-1) 
