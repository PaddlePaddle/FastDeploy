# YOLOv5Seg准备部署模型

- YOLOv5Seg v7.0部署模型实现来自[YOLOv5](https://github.com/ultralytics/yolov5/tree/v7.0),和[基于COCO的预训练模型](https://github.com/ultralytics/yolov5/releases/tag/v7.0)
  - （1）[官方库](https://github.com/ultralytics/yolov5/releases/tag/v7.0)提供的*.onnx可直接进行部署；
  - （2）开发者基于自己数据训练的YOLOv5Seg v7.0模型，可使用[YOLOv5](https://github.com/ultralytics/yolov5)中的`export.py`导出ONNX文件后，完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5Seg导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOv5n-seg](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-seg.onnx) | 7.7MB | 27.6% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5s-seg](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-seg.onnx) | 30MB | 37.6% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5m-seg](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5m-seg.onnx) | 84MB | 45.0% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5l-seg](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5l-seg.onnx) | 183MB | 49.0% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5x-seg](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5x-seg.onnx) | 339MB | 50.7% | 此模型文件来源于[YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0) 编写
