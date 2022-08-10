# YOLOv6准备部署模型

## 模型版本说明

- [YOLOv6 v0.1.0](https://github.com/meituan/YOLOv6/releases/download/0.1.0)
  - （1）[链接中](https://github.com/meituan/YOLOv6/releases/download/0.1.0)的*.onnx可直接进行部署；


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv6导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv6s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s.onnx) | 66MB | 43.1% |
| [YOLOv6s_640](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s-640x640.onnx) | 66MB | 43.1% |

nanodet-plus-m_320.onnx  nanodet-plus-m_320-sim.onnx

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
