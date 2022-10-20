# RobustVideoMatting 模型部署

## 模型版本说明

- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/commit/81a1093)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [RobustVideoMatting 模型](https://github.com/PeterL1n/RobustVideoMatting)


## 下载预训练模型

为了方便开发者的测试，下面提供了PP-Matting导出的各系列模型，开发者可直接下载使用。

其中精度指标来源于PP-Matting中对各模型的介绍(未提供精度数据)，详情各参考PP-Matting中的说明。


| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [rvm_mobilenetv3_fp32.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_fp32.onnx) | 15MB | - |
| [rvm_resnet50_fp32.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_resnet50_fp32.onnx) | 103MB | - |
| [rvm_mobilenetv3_trt.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_trt.onnx) | 15MB | - |
| [rvm_resnet50_trt.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/rvm_resnet50_trt.onnx) | 103MB | - |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
