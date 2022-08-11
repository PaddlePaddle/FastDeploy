# YOLOX准备部署模型

## 模型版本说明
- YOLOX部署实现来自[YOLOX v0.1.1分支](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.1rc0)，基于[coco的预训练模型](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)。

  - （1）[预训练模型](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)中的*.pth通过导出ONNX模型操作后，可进行部署;*.onnx、*.trt和*.pose模型不支持部署；
  - （2）开发者基于自己数据训练的YOLOX v0.1.1模型，可按照导出ONNX模型后，完成部署。

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOX导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOX-s](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s.onnx) | 35MB | 40.5% |




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
