[English](README.md) | 简体中文

# PaddleDetection RKNPU2部署示例

## 支持模型列表

目前FastDeploy使用RKNPU2支持如下PaddleDetection模型的部署:

- Picodet
- PPYOLOE
- YOLOV8

## 准备PaddleDetection部署模型以及转换模型

RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:

* Paddle动态图模型转换为ONNX模型，请参考[PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md)
,注意在转换时请设置**export.nms=True**.
* ONNX模型转换RKNN模型的过程，请参考[转换文档](../../../../../docs/cn/faq/rknpu2/export.md)进行转换。

## 模型转换example

- [Picodet RKNPU2模型转换文档](./picodet.md)
- [YOLOv8 RKNPU2模型转换文档](./yolov8.md)


## 其他链接

- [Cpp部署](./cpp)
- [Python部署](./python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
