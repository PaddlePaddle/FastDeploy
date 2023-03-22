# YOLOv7Face准备部署模型

- YOLOv7Face部署模型实现来自[YOLOv7Face](https://github.com/derronqi/yolov7-face),和[基于WiderFace的预训练模型](https://github.com/derronqi/yolov7-face)
  - （1）[官方库](https://github.com/derronqi/yolov7-face)中提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的YOLOv7Face模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 导出ONNX模型

访问[YOLOv7Face](https://github.com/derronqi/yolov7-face)官方github库，按照指引下载安装，下载`.pt` 模型，利用 `export.py` 得到`onnx`格式文件。

* 下载yolov7模型文件

| Method           |  Test Size | Easy  | Medium | Hard  | FLOPs (B) @640 | Link  |
| -----------------| ---------- | ----- | ------ | ----- | -------------- | ----- |
|  yolov7-lite-t   | 640        | 88.7  | 85.2   | 71.5  |  0.8           | [google](https://drive.google.com/file/d/1HNXd9EdS-BJ4dk7t1xJDFfr1JIHjd5yb/view?usp=sharing) |
| yolov7-lite-s    | 640        | 92.7  | 89.9   | 78.5  |  3.0           | [google](https://drive.google.com/file/d/1MIC5vD4zqRLF_uEZHzjW_f-G3TsfaOAf/view?usp=sharing) |
| yolov7-tiny      | 640        | 94.7  | 92.6   | 82.1  |  13.2          | [google](https://drive.google.com/file/d/1Mona-I4PclJr5mjX1qb8dgDeMpYyBcwM/view?usp=sharing) |
| yolov7s          | 640        | 94.8  | 93.1   | 85.2  |  16.8          | [google](https://drive.google.com/file/d/1_ZjnNF_JKHVlq41EgEqMoGE2TtQ3SYmZ/view?usp=sharing) |
| yolov7           | 640        | 96.9  | 95.5   | 88.0  |  103.4         | [google](https://drive.google.com/file/d/1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo/view?usp=sharing) |
| yolov7+TTA       | 640        | 97.2  | 95.8   | 87.7  |  103.4         | [google](https://drive.google.com/file/d/1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo/view?usp=sharing) |
| yolov7-w6        | 960        | 96.4  | 95.0   | 88.3  |  89.0          | [google](https://drive.google.com/file/d/1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS/view?usp=sharing) |
| yolov7-w6+TTA    | 1280       | 96.9  | 95.8   | 90.4  |  89.0          | [google](https://drive.google.com/file/d/1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS/view?usp=sharing) |

* 导出onnx格式文件
  ```bash
  python ./models/export.py --weights yolov7-tiny-face.pt --grid --simplify --img-size 640 640
  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOv7Face导出的各系列模型，开发者可直接下载使用。
| 模型                                                               | 大小    |
|:---------------------------------------------------------------- |:----- |
| [yolov7-lite-e](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-lite-e.onnx) | 3.2MB |
| [yolov7-tiny-face](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-tiny-face.onnx) | 30.3MB |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOv7Face](https://github.com/derronqi/yolov7-face) 编写
