English | [简体中文](README_CN.md)

# YOLOv8 Ready-to-deploy Model

- The deployment of the YOLOv8 model is based on [YOLOv8](https://github.com/ultralytics/ultralytics) and [Pre-trained Model Based on COCO](https://github.com/ultralytics/ultralytics)
  - （1）The *.onnx provided by [Official Repository](https://github.com/ultralytics/ultralytics) can be deployed directly；
  - （2）The YOLOv8 model trained by personal data should employ `export.py` in [YOLOv8](https://github.com/ultralytics/ultralytics) to export the ONNX files for deployment.

## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOv8 are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy  | Note |
|:---------------------------------------------------------------- |:----- |:----- |:---- |
| [YOLOv8n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8n.onnx) | 12.1MB | 37.3% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8s.onnx) | 42.6MB | 44.9% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8m](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8m.onnx) | 98.8MB | 50.2% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8l](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8l.onnx) | 166.7MB | 52.9% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |
| [YOLOv8x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8x.onnx) | 260.3MB | 53.9% | This model file is sourced from [YOLOv8](https://github.com/ultralytics/ultralytics)，GPL-3.0 License |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Release Note

- Document and code are based on [YOLOv8](https://github.com/ultralytics/ultralytics)
