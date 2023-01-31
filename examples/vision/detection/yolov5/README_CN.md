[English](README.md) | 简体中文
# YOLOv5准备部署模型

- YOLOv5 v7.0部署模型实现来自[YOLOv5](https://github.com/ultralytics/yolov5/tree/v7.0),和[基于COCO的预训练模型](https://github.com/ultralytics/yolov5/releases/tag/v7.0)
  - （1）[官方库](https://github.com/ultralytics/yolov5/releases/tag/v7.0)提供的*.onnx可直接进行部署；
  - （2）开发者基于自己数据训练的YOLOv5 v7.0模型，可使用[YOLOv5](https://github.com/ultralytics/yolov5)中的`export.py`导出ONNX文件后，完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度  | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [YOLOv5n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n.onnx) | 7.6MB | 28.0% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx) | 28MB | 37.4% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5m](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5m.onnx) | 82MB | 45.4% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5l](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5l.onnx) | 178MB | 49.0% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5x.onnx) | 332MB | 50.7% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
- [服务化部署](serving)

## 版本说明

- 本版本文档和代码基于[YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0) 编写
