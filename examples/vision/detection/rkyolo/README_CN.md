[English](README.md) | 简体中文
# RKYOLO准备部署模型

RKYOLO参考[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo)的代码
对RKYOLO系列模型进行了封装，目前支持RKYOLOV5系列模型的部署。

## 支持模型列表

FastDeploy目前支持以下三个模型的部署:

* RKYOLOV5
* RKYOLOX
* RKYOLOv7

为了方便大家测试，我们提供了三个转换过后的模型，大家可以直接下载使用。
如果你有转换模型的需求，请参考[RKNN_model_convert](https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo/RKNN_model_convert)

| 模型名称               | 下载地址                                                                |
|--------------------|---------------------------------------------------------------------|
| yolov5-s-relu-int8 | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolov5-s-relu.zip |
| yolov7-tiny-int8   | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolov7-tiny.zip   |
| yolox-s-int8       | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolox-s.zip       |



## 其他链接
- [Cpp部署](./cpp)
- [Python部署](./python)
- [视觉模型预测结果](../../../../docs/api/vision_results/)
