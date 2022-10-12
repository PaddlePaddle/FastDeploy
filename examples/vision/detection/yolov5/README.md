# YOLOv5准备部署模型

- YOLOv5 v6.0部署模型实现来自[YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.0),和[基于COCO的预训练模型](https://github.com/ultralytics/yolov5/releases/tag/v6.0)
  - （1）[官方库](https://github.com/ultralytics/yolov5/releases/tag/v6.0)提供的*.onnx可直接进行部署；
  - （2）开发者基于自己数据训练的YOLOv5 v6.0模型，可使用[YOLOv5](https://github.com/ultralytics/yolov5)中的`export.py`导出ONNX文件后，完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv5n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n.onnx) | 1.9MB | 28.4% |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx) | 7.2MB | 37.2% |
| [YOLOv5m](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5m.onnx) | 21.2MB | 45.2% |
| [YOLOv5l](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5l.onnx) | 46.5MB | 48.8% |
| [YOLOv5x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5x.onnx) | 86.7MB | 50.7% |




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
- [服务化部署](serving)

## 版本说明

- 本版本文档和代码基于[YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/v6.0) 编写
