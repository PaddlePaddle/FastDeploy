[English](README.md) | 简体中文
# YOLOv8准备部署模型

- YOLOv8部署模型实现来自[YOLOv8](https://github.com/ultralytics/ultralytics),和[基于COCO的预训练模型](https://github.com/ultralytics/ultralytics)
  - （1）[官方库](https://github.com/ultralytics/ultralytics)提供的*.onnx可直接进行部署；
  - （2）开发者基于自己数据训练的YOLOv8模型，可使用[YOLOv8](https://github.com/ultralytics/ultralytics)中的`export.py`导出ONNX文件后，完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv8导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度  | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [YOLOv8n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8n.onnx) | 12.1MB | 37.3% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8s.onnx) | 42.6MB | 44.9% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8m](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8m.onnx) | 98.8MB | 50.2% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8l](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8l.onnx) | 166.7MB | 52.9% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8x.onnx) | 260.3MB | 53.9% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[YOLOv8](https://github.com/ultralytics/ultralytics) 编写
