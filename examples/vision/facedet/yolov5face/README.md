# YOLOv5Face准备部署模型

## 模型版本说明

- [YOLOv5Face](https://github.com/deepcam-cn/yolov5-face/commit/4fd1ead)
  - （1）[链接中](https://github.com/deepcam-cn/yolov5-face/commit/4fd1ead)的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的YOLOv5Face模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 导出ONNX模型

访问[YOLOv5Face](https://github.com/deepcam-cn/yolov5-face)官方github库，按照指引下载安装，下载`yolov5s-face.pt` 模型，利用 `export.py` 得到`onnx`格式文件。

* 下载yolov5face模型文件
  ```
  Link: https://pan.baidu.com/s/1fyzLxZYx7Ja1_PCIWRhxbw Link: eq0q  
  https://drive.google.com/file/d/1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO/view?usp=sharing
  ```

* 导出onnx格式文件
  ```bash
  PYTHONPATH=. python export.py --weights weights/yolov5s-face.pt --img_size 640 640 --batch_size 1  
  ```
* onnx模型简化(可选)
  ```bash
  onnxsim yolov5s-face.onnx yolov5s-face.onnx
  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv5Face导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv5s-Face](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-face.onnx) | 30MB | 94.3 |
| [YOLOv5s-Face-bak](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5face-s-640x640.bak.onnx) | 30MB | -|
| [YOLOv5l-Face](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5face-l-640x640.onnx ) | 181MB | 95.8 |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOv5Face CommitID:4fd1ead](https://github.com/deepcam-cn/yolov5-face/commit/4fd1ead) 编写
