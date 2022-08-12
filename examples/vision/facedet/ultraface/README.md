# UltraFace准备部署模型


- [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/commit/dffdddd)
  - （1）[官方库中提供的](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/)的*.onnx可下载, 也可以通过下面模型链接下载并进行部署
  - （2）开发者自己训练的模型，导出ONNX模型后，参考[详细部署文档](#详细部署文档)完成部署。



## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了UltraFace导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [RFB-320](https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320.onnx) | 1.3MB | - |
| [RFB-320-sim](https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320-sim.onnx) | 1.2MB | -|



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[UltraFace CommitID:dffdddd](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/commit/dffdddd) 编写
