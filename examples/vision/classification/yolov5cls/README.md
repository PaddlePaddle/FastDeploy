English | [简体中文](README_CN.md)

# YOLOv5Cls Ready-to-deploy Model

- YOLOv5Cls v6.2 model deployment is based on [YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.2) and [Pre-trained Models on ImageNet](https://github.com/ultralytics/yolov5/releases/tag/v6.2).
  - （1）The *-cls.pt model provided by [Official Repository](https://github.com/ultralytics/yolov5/releases/tag/v6.2) can export the ONNX file using `export.py` in [YOLOv5](https://github.com/ultralytics/yolov5), then deployment can be conducted;
  - （2）The YOLOv5Cls v6.2 Model trained by personal data should export the ONNX file using `export.py` in [YOLOv5](https://github.com/ultralytics/yolov5).


## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOv5Cls are provided below. Developers can download them directly. (The model accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy(top1)  | Accuracy(top5)    |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOv5n-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-cls.onnx) | 9.6MB | 64.6% | 85.4% |
| [YOLOv5s-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-cls.onnx) | 21MB | 71.5% | 90.2% |
| [YOLOv5m-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5m-cls.onnx) | 50MB | 75.9% | 92.9% |
| [YOLOv5l-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5l-cls.onnx) | 102MB | 78.0% | 94.0% |
| [YOLOv5x-cls](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5x-cls.onnx) | 184MB | 79.0% | 94.4% |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Release Note

- Document and code are based on [YOLOv5 v6.2](https://github.com/ultralytics/yolov5/tree/v6.2).
