[English](README.md) | 简体中文
# YOLOv5Cls准备部署模型

- YOLOv5Cls v6.2部署模型实现来自[YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.2),和[基于ImageNet的预训练模型](https://github.com/ultralytics/yolov5/releases/tag/v6.2)
  - （1）[官方库](https://github.com/ultralytics/yolov5/releases/tag/v6.2)提供的*-cls.pt模型，使用[YOLOv5](https://github.com/ultralytics/yolov5)中的`export.py`导出ONNX文件后，可直接进行部署；
  - （2）开发者基于自己数据训练的YOLOv5Cls v6.2模型，可使用[YOLOv5](https://github.com/ultralytics/yolov5)中的`export.py`导出ONNX文件后，完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5Cls导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度(top1)  | 精度(top5)    |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOv5n-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-cls.onnx) | 9.6MB | 64.6% | 85.4% |
| [YOLOv5s-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-cls.onnx) | 21MB | 71.5% | 90.2% |
| [YOLOv5m-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5m-cls.onnx) | 50MB | 75.9% | 92.9% |
| [YOLOv5l-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5l-cls.onnx) | 102MB | 78.0% | 94.0% |
| [YOLOv5x-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5x-cls.onnx) | 184MB | 79.0% | 94.4% |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[YOLOv5 v6.2](https://github.com/ultralytics/yolov5/tree/v6.2) 编写
