[English](README.md) | 简体中文
# ScaledYOLOv4准备部署模型

- ScaledYOLOv4部署实现来自[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)的代码，和[基于COCO的预训练模型](https://github.com/WongKinYiu/ScaledYOLOv4)。

  - （1）[官方库](https://github.com/WongKinYiu/ScaledYOLOv4)提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）自己数据训练的ScaledYOLOv4模型，按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)操作后，参考[详细部署文档](#详细部署文档)完成部署。


## 导出ONNX模型


  访问[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)官方github库，按照指引下载安装，下载`scaledyolov4.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。如果您导出的`onnx`模型出现问题，可以参考[ScaledYOLOv4#401](https://github.com/WongKinYiu/ScaledYOLOv4/issues/401)的解决办法

  ```bash
  #下载ScaledYOLOv4模型文件
  Download from the goole drive https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz/view?usp=sharing

  # 导出onnx格式文件
  python models/export.py  --weights PATH/TO/scaledyolov4-xx.pt --img-size 640
  ```


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了ScaledYOLOv4导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOR-P6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-p6-paper-541-1280-1280.onnx) | 143MB | 54.1% | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-W6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-w6-paper-555-1280-1280.onnx) | 305MB | 55.5% | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-E6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-e6-paper-564-1280-1280.onnx ) | 443MB | 56.4% | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-570-1280-1280.onnx) | 580MB | 57.0% | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-573-1280-1280.onnx) | 580MB | 57.3% | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-P6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-p6-paper-541-640-640.onnx) | 143MB | - | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-W6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-w6-paper-555-640-640.onnx) | 305MB | - | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-E6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-e6-paper-564-640-640.onnx ) | 443MB | - | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-570-640-640.onnx) | 580MB | - | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-573-640-640.onnx) | 580MB | - | 此模型文件来源于[YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[ScaledYOLOv4 CommitID: 6768003](https://github.com/WongKinYiu/ScaledYOLOv4/commit/676800364a3446900b9e8407bc880ea2127b3415) 编写
