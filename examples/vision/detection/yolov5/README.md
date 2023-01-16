English | [简体中文](README_CN.md)

# YOLOv5 Ready-to-deploy Model

- The deployment of the YOLOv5 v7.0 model is based on [YOLOv5](https://github.com/ultralytics/yolov5/tree/v7.0) and [Pre-trained Model Based on COCO](https://github.com/ultralytics/yolov5/releases/tag/v7.0)
  - （1）The *.onnx provided by [Official Repository](https://github.com/ultralytics/yolov5/releases/tag/v7.0) can be deployed directly；
  - （2）The YOLOv5 v7.0 model trained by personal data should employ `export.py` in [YOLOv5](https://github.com/ultralytics/yolov5) to export the ONNX files for deployment.

## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOv5 are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy  | Note |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [YOLOv5n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n.onnx) | 7.6MB | 28.0% | This model file is sourced from [YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx) | 28MB | 37.4% | This model file is sourced from [YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5m](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5m.onnx) | 82MB | 45.4% | This model file is sourced from [YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5l](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5l.onnx) | 178MB | 49.0% | This model file is sourced from [YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |
| [YOLOv5x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5x.onnx) | 332MB | 50.7% | This model file is sourced from [YOLOv5](https://github.com/ultralytics/yolov5)，GPL-3.0 License |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)
- [Serving Deployment](serving)

## Release Note

- Document and code are based on [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0)
