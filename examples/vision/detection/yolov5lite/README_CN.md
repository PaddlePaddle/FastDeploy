[English](README.md) | 简体中文
# YOLOv5Lite准备部署模型

- YOLOv5Lite部署实现来自[YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)
代码，和[基于COCO的预训练模型](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)。

  - （1）[官方库](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）自己数据训练的YOLOv5Lite模型，按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)操作后，参考[详细部署文档](#详细部署文档)完成部署。


## 导出ONNX模型

- 自动获取
  访问[YOLOv5Lite](https://github.com/ppogg/YOLOv5-Lite)
官方github库，按照指引下载安装，下载`yolov5-lite-xx.onnx` 模型(Tips：官方提供的ONNX文件目前是没有decode模块的)
  ```bash
  #下载yolov5-lite模型文件(.onnx)
  Download from https://drive.google.com/file/d/1bJByk9eoS6pv8Z3N4bcLRCV3i7uk24aU/view
  官方Repo也支持百度云下载
  ```

- 手动获取

  访问[YOLOv5Lite](https://github.com/ppogg/YOLOv5-Lite)
官方github库，按照指引下载安装，下载`yolov5-lite-xx.pt` 模型，利用 `export.py` 得到`onnx`格式文件。

  - 导出含有decode模块的ONNX文件

  首先需要参考[YOLOv5-Lite#189](https://github.com/ppogg/YOLOv5-Lite/pull/189)的解决办法，修改代码。

  ```bash
  #下载yolov5-lite模型文件(.pt)
  Download from https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view
  官方Repo也支持百度云下载

  # 导出onnx格式文件
  python export.py --grid --dynamic --concat --weights PATH/TO/yolov5-lite-xx.pt


  ```
  - 导出无decode模块的ONNX文件(不需要修改代码)

  ```bash
  #下载yolov5-lite模型文件
  Download from https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view
  官方Repo也支持百度云下载

  # 导出onnx格式文件
  python export.py --grid --dynamic --weights PATH/TO/yolov5-lite-xx.pt

  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5Lite导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOv5Lite-e](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-e-sim-320.onnx) | 3.1MB | 35.1% | 此模型文件来源于[YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |
| [YOLOv5Lite-s](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-s-sim-416.onnx) | 6.3MB | 42.0% | 此模型文件来源于[YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |
| [YOLOv5Lite-c](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-c-sim-512.onnx) | 18MB | 50.9% | 此模型文件来源于[YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |
| [YOLOv5Lite-g](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-g-sim-640.onnx) | 21MB | 57.6% | 此模型文件来源于[YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOv5-Lite v1.4](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4) 编写
