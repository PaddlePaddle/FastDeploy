# MODNet准备部署模型

- [MODNet](https://github.com/ZHKKKe/MODNet/commit/28165a4)
  - （1）[官方库](https://github.com/ZHKKKe/MODNet/)中提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的MODNet模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 导出ONNX模型


访问[MODNet](https://github.com/ZHKKKe/MODNet)官方github库，按照指引下载安装，下载模型文件，利用 `onnx/export_onnx.py` 得到`onnx`格式文件。

* 导出onnx格式文件
  ```bash
  python -m onnx.export_onnx \
    --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
    --output-path=pretrained/modnet_photographic_portrait_matting.onnx
  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了MODNet导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [PPMatting](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting.onnx) | 25MB | - |
TODO links




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[MODNet CommitID:28165a4](https://github.com/ZHKKKe/MODNet/commit/28165a4) 编写
