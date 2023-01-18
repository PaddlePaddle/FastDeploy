English | [简体中文](README_CN.md)
# YOLOX Ready-to-deploy Model


- The YOLOX deployment is based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.1rc0) and [coco's pre-trained models](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0).

  - （1）The *.pth provided by [Official Repository](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) should export the ONNX model to complete the deployment;
  - （2）The YOLOX model trained by personal data should export the ONNX model. Refer to [Detailed Deployment Documents](#Detailed-Deployment-Documents)  to complete the deployment.



## Download Pre-trained ONNX Models


For developers' testing, models exported by YOLOX are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOX-s](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s.onnx) | 35MB | 39.6% |
| [YOLOX-m](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_m.onnx) | 97MB | 46.4.5% |
| [YOLOX-l](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_l.onnx) | 207MB | 50.0% |
| [YOLOX-x](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_x.onnx) | 378MB | 51.2% |




## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- Document and code are based on [YOLOX v0.1.1 version](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.1rc0) 
