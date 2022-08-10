# RetinaFace准备部署模型

## 模型版本说明

- [ArcFace CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5)
  - （1）[链接中](https://github.com/deepinsight/insightface/commit/babb9a5)的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的RetinaFace CommitID:b984b4b模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 导出ONNX模型

访问[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)官方github库，按照指引下载安装，下载pt模型文件，利用 `torch2onnx.py` 得到`onnx`格式文件。

* 下载ArcFace模型文件
  ```
  Link: https://pan.baidu.com/share/init?surl=CL-l4zWqsI1oDuEEYVhj-g code: e8pw  
  ```

* 导出onnx格式文件
  ```bash
  PYTHONPATH=. python ./torch2onnx.py ms1mv3_arcface_r100_fp16/backbone.pth --output ms1mv3_arcface_r100.onnx --network r100 --simplify 1
  ```

## 下载预训练ONNX模型

<!-- 为了方便开发者的测试，下面提供了RetinaFace导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [RetinaFace_mobile0.25-640](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-640-640.onnx) | 1.7MB | - |
| [RetinaFace_mobile0.25-720](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-720-1080.onnx) | 1.7MB | -|
| [RetinaFace_resnet50-640](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_resnet50-720-1080.onnx) | 105MB | - |
| [RetinaFace_resnet50-720](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_resnet50-640-640.onnx) | 105MB | - | -->

todo


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
