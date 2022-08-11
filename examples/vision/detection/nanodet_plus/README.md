# NanoDetPlus准备部署模型

## 模型版本说明

- NanoDetPlus部署实现来自[NanoDetPlus v1.0.0-alpha-1](https://github.com/RangiLyu/nanodet/tree/v1.0.0-alpha-1) 代码分支，基于coco的[预训练模型](https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1)。

  - （1）[预训练模型](https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1)的*.onnx可直接进行部署；
  - （2）自己训练的模型，导出ONNX模型后，参考[详细部署教程](#详细部署文档)完成部署。
  
## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了NanoDetPlus导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [NanoDetPlus_320](https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx ) | 4.6MB | 27.0% |
| [NanoDetPlus_320_sim](https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320-sim.onnx) | 4.6MB | 27.0% |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
