# YOLOv7部署

## 版本依赖

- [YOLOv7 0.1](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)


## 导出ONNX模型

```
#下载yolov7模型文件
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# 导出onnx格式文件 (Tips: 对应 YOLOv7 release v0.1 代码)
python models/export.py --grid --dynamic --weights PATH/TO/yolov7.pt

# 如果您的代码版本中有支持NMS的ONNX文件导出，请使用如下命令导出ONNX文件(请暂时不要使用 "--end2end"，我们后续将支持带有NMS的ONNX模型的部署)
python export.py --grid --dynamic --weights PATH/TO/yolov7.pt

# 移动onnx文件到demo目录
cp PATH/TO/yolov7.onnx PATH/TO/model_zoo/vision/yolov7/
```

## 预训练模型集合

为了方便开发者的测试，下面提供了YOLOv7导出的各系列模型，开发者可直接下载使用。

| 模型 | 大小 | 精度 |
| :--- | :--- | :--- |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7.onnx) | 141MB | 51.4% |
| [YOLOv7-x] | 10MB | 51.4% |


## 其它文档

- [Python部署](python)
- [C++部署](cpp)
