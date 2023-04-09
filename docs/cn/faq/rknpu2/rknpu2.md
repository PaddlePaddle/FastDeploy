[English](../../../en/faq/rknpu2/rknpu2.md) | 中文
# RKNPU2概述

## 安装环境
RKNPU2模型导出只支持在x86Linux平台上进行导出，安装流程请参考[RKNPU2模型导出环境配置文档](./environment.md)

## ONNX模型转换为RKNN模型
ONNX模型不能直接调用RK芯片中的NPU进行运算，需要把ONNX模型转换为RKNN模型，具体流程请查看[RKNPU2转换文档](./export.md)

## RKNPU2已经支持的模型列表

FastDeploy在RK3588s上进行了测试，测试环境如下:

* 设备型号: RK3588-s
* NPU均使用单核进行测试

以下环境测试的速度均为端到端测试速度根据芯片体质的不同，速度会上下有所浮动，仅供参考。

| 任务场景                 | 模型及其example                                                                                       | 模型版本                     | 是否量化 | RKNN速度(ms) |
|----------------------|---------------------------------------------------------------------------------------------------|--------------------------|------|------------|
| Classification       | [ResNet](../../../../examples/vision/classification/paddleclas/rknpu2/README.md)                  | ResNet50_vd              | 否    | 33         |
| Detection            | [Picodet](../../../../examples/vision/detection/paddledetection/rknpu2/README.md)                 | Picodet-s                | 否    | 112        |
| Detection            | [PaddleDetection Yolov8](../../../../examples/vision/detection/paddledetection/rknpu2/README.md)  | yolov8-n                 | 否    | 100        |
| Detection            | [PPYOLOE](../../../../examples/vision/detection/paddledetection/rknpu2/README.md)                 | ppyoloe-s(int8)          | 是    | 141        |
| Detection            | [RKYOLOV5](../../../../examples/vision/detection/rkyolo/README.md)                                | YOLOV5-S-Relu(int8)      | 是    | 57         |
| Detection            | [RKYOLOX](../../../../examples/vision/detection/rkyolo/README.md)                                 | yolox-s                  | 是    | 130        |
| Detection            | [RKYOLOV7](../../../../examples/vision/detection/rkyolo/README.md)                                | yolov7-tiny              | 是    | 58         |
| Segmentation         | [Unet](../../../../examples/vision/segmentation/paddleseg/rockchip/rknpu2/README.md)              | Unet-cityscapes          | 否    | -          |
| Segmentation         | [PP-HumanSegV2Lite](../../../../examples/vision/segmentation/paddleseg/rockchip/rknpu2/README.md) | portrait(int8)           | 是    | 43         |
| Segmentation         | [PP-HumanSegV2Lite](../../../../examples/vision/segmentation/paddleseg/rockchip/rknpu2/README.md) | human(int8)              | 是    | 43         |
| Face Detection       | [SCRFD](../../../../examples/vision/facedet/scrfd/rknpu2/README.md)                               | SCRFD-2.5G-kps-640(int8) | 是    | 42         |
| Face FaceRecognition | [InsightFace](../../../../examples/vision/faceid/insightface/rknpu2/README_CN.md)                 | ms1mv3_arcface_r18(int8) | 是    | 12         |

## 预编译库下载

为了方便大家进行开发，这里提供1.0.2版本的FastDeploy给大家使用

- [FastDeploy RK356X c++ SDK](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-rk356X-1.0.2.tgz)
- [FastDeploy RK3588 c++ SDK](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-rk3588-1.0.2.tgz)
