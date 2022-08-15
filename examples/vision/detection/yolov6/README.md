# YOLOv6准备部署模型


- YOLOv6 部署实现来自[YOLOv6](https://github.com/meituan/YOLOv6/releases/tag/0.1.0)，和[基于coco的预训练模型](https://github.com/meituan/YOLOv6/releases/tag/0.1.0)。

  - （1）[官方库](https://github.com/meituan/YOLOv6/releases/tag/0.1.0)提供的*.onnx可直接进行部署；
  - （2）开发者自己训练的模型，导出ONNX模型后，参考[详细部署文档](#详细部署文档)完成部署。



## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv6导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv6s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s.onnx) | 66MB | 43.1% |
| [YOLOv6s_640](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s-640x640.onnx) | 66MB | 43.1% |
| [YOLOv6t](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6t.onnx) | 58MB | 41.3% |
| [YOLOv6n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6n.onnx) | 17MB | 35.0% |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOv6 0.1.0版本](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) 编写
