[English](README.md) | 简体中文
# YOLOv7End2EndORT 准备部署模型

YOLOv7End2EndORT 部署实现来自[YOLOv7](https://github.com/WongKinYiu/yolov7/tree/v0.1)分支代码，和[基于COCO的预训练模型](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)。注意，YOLOv7End2EndORT是专门用于推理YOLOv7中导出模型带[ORT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L87) 版本的End2End模型，不带nms的模型推理请使用YOLOv7类，而 [TRT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L111) 版本的End2End模型请使用YOLOv7End2EndTRT进行推理。

  - （1）[官方库](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；*.trt和*.pose模型不支持部署；
  - （2）自己数据训练的YOLOv7模型，按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)操作后，参考[详细部署文档](#详细部署文档)完成部署。


## 导出ONNX模型

```bash
# 下载yolov7模型文件
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# 导出带ORT_NMS的onnx格式文件 (Tips: 对应 YOLOv7 release v0.1 代码)
python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
# 导出其他模型的命令类似 将yolov7.pt替换成 yolov7x.pt yolov7-d6.pt yolov7-w6.pt ...
```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv7End2EndORT导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [yolov7-end2end-ort-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-end2end-ort-nms.onnx) | 141MB | 51.4% | 此模型文件来源于[YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7x-end2end-ort-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7x-end2end-ort-nms.onnx) | 273MB | 53.1% | 此模型文件来源于[YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-w6-end2end-ort-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-w6-end2end-ort-nms.onnx) | 269MB | 54.9% | 此模型文件来源于[YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-e6-end2end-ort-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6-end2end-ort-nms.onnx) | 372MB | 56.0% | 此模型文件来源于[YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-d6-end2end-ort-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-d6-end2end-ort-nms.onnx) | 511MB | 56.6% | 此模型文件来源于[YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-e6e-end2end-ort-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6e-end2end-ort-nms.onnx) | 579MB | 56.8% | 此模型文件来源于[YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOv7 0.1](https://github.com/WongKinYiu/yolov7/tree/v0.1) 编写
