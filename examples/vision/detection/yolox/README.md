# YOLOX准备部署模型


- YOLOX部署实现来自[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.1rc0)，基于[coco的预训练模型](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)。

  - （1）[官方库](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)提供中的*.pth通过导出ONNX模型操作后，可进行部署;
  - （2）开发者自己训练的模型，导出ONNX模型后，参考[详细部署文档](#详细部署文档)完成部署。


## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了YOLOX导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOX-s](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s.onnx) | 35MB | 39.6% |
| [YOLOX-m](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_m.onnx) | 97MB | 46.4.5% |
| [YOLOX-l](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_tiny.onnx) | 207MB | 50.0% |
| [YOLOX-x](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_x.onnx) | 378MB | 51.2% |




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[YOLOX v0.1.1版本](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.1rc0) 编写
