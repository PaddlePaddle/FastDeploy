English | [简体中文](README.md)

# YOLOv6 Ready-to-deploy Model


- The YOLOv6 deployment is based on [YOLOv6](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) and [Pre-trained Model Based on COCO](https://github.com/meituan/YOLOv6/releases/tag/0.1.0).

  - （1）The *.onnx provided by [Official Repository](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) can directly conduct deployemnt；
  - （2）Personal models trained by developers should export the ONNX model. Refer to [Detailed Deployment Documents](#详细部署文档) to complete the deployment.



## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOv6 are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv6s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s.onnx) | 66MB | 43.1% |
| [YOLOv6s_640](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s-640x640.onnx) | 66MB | 43.1% |
| [YOLOv6t](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6t.onnx) | 58MB | 41.3% |
| [YOLOv6n](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6n.onnx) | 17MB | 35.0% |



## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployement](cpp)


## Release Note

- Document and code are based on [YOLOv6 0.1.0 version](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) 