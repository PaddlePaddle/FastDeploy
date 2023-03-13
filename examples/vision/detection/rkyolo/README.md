English | [简体中文](README_CN.md)

# RKYOLO Ready-to-deploy Model

RKYOLO models are encapsulated with reference to the code of [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo). Now we support the deployment of RKYOLOV5 models. 

## List of Supported Models

FastDeploy currently supports the deployment of the following three models: 

- RKYOLOV5
- RKYOLOX
- RKYOLOv7

For people’s testing, we provide three converted models that allow downloading and use. If you need to convert models, refer to [RKNN_model_convert](https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo/RKNN_model_convert).

| Model Name           | Download Address                                                     |
| ------------------ | ------------------------------------------------------------ |
| yolov5-s-relu-int8 | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolov5-s-relu.zip |
| yolov7-tiny-int8   | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolov7-tiny.zip |
| yolox-s-int8       | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolox-s.zip |


## Other Links
- [Cpp deployment](./cpp)
- [Python deployment](./python)
- [Vision model predicting results](../../../../docs/api/vision_results/)
