# YOLOv5准备部署模型

- YOLOv5 v6.2部署模型实现来自[YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.2)
    - （1）[官方库](https://github.com/ultralytics/yolov5/releases/tag/v6.2)提供的*.onnx可直接进行转换模型；
    - （2）开发者基于自己数据训练的YOLOv5 v6.2模型，可使用[YOLOv5](https://github.com/ultralytics/yolov5)中的`export.py`导出ONNX文件(注意op版本请设置为11)。



## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5s_v6.2ONNX格式的模型，开发者可直接下载使用。

| 模型                                                                 | 大小     | 精度    |
|:-------------------------------------------------------------------|:-------|:------|
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx) | 7.2MB  | 37.2% |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[YOLOv5 v6.2](https://github.com/ultralytics/yolov5/tree/v6.0) 编写
