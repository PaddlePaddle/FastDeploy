简体中文 | [English](README_EN.md)
# YOLOv7准备部署模型

- YOLOv7部署实现来自[YOLOv7](https://github.com/WongKinYiu/yolov7/tree/v0.1)分支代码，和[基于COCO的预训练模型](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)。

  - （1）[官方库](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；*.trt和*.pose模型不支持部署；
  - （2）自己数据训练的YOLOv7模型，按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)操作后，参考[详细部署文档](#详细部署文档)完成部署。




## 导出ONNX模型

```bash
# 下载yolov7模型文件
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# 导出onnx格式文件 (Tips: 对应 YOLOv7 release v0.1 代码)
python models/export.py --grid --dynamic --weights PATH/TO/yolov7.pt

# 如果您的代码版本中有支持NMS的ONNX文件导出，请使用如下命令导出ONNX文件(请暂时不要使用 "--end2end"，我们后续将支持带有NMS的ONNX模型的部署)
python models/export.py --grid --dynamic --weights PATH/TO/yolov7.pt


```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv7导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7.onnx) | 141MB | 51.4% |
| [YOLOv7x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7x.onnx) | 273MB | 53.1% |
| [YOLOv7-w6](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-w6.onnx) | 269MB | 54.9% |
| [YOLOv7-e6](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6.onnx) | 372MB | 56.0% |
| [YOLOv7-d6](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-d6.onnx) | 511MB | 56.6% |
| [YOLOv7-e6e](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6e.onnx) | 579MB | 56.8% |




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOv7 0.1](https://github.com/WongKinYiu/yolov7/tree/v0.1) 编写
