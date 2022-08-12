# MODNet准备部署模型

## 模型版本说明

- [MODNet](https://github.com/ZHKKKe/MODNet/commit/28165a4)
  - （1）[链接中](https://github.com/ZHKKKe/MODNet/commit/28165a4)的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的MODNet CommitID:b984b4b模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 导出ONNX模型


访问[MODNet](https://github.com/ZHKKKe/MODNet)官方github库，按照指引下载安装，下载模型文件，利用 `onnx/export_onnx.py` 得到`onnx`格式文件。

* 导出onnx格式文件
  ```bash
  python -m onnx.export_onnx \
    --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
    --output-path=pretrained/modnet_photographic_portrait_matting.onnx
  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了MODNet导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [modnet_photographic](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic__portrait_matting.onnx) | 25MB | - |
| [modnet_webcam](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting.onnx) | 25MB | -|
| [modnet_photographic_256](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting-256x256.onnx) | 25MB | - |
| [modnet_webcam_256](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting-256x256.onnx) | 25MB | - |
| [modnet_photographic_512](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting-512x512.onnx) | 25MB  | - |
| [modnet_webcam_512](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting-512x512.onnx) | 25MB | - |
| [modnet_photographic_1024](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting-1024x1024.onnx) | 25MB | - |
| [modnet_webcam_1024](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting-1024x1024.onnx) | 25MB | -|




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[MODNet CommitID:28165a4](https://github.com/ZHKKKe/MODNet/commit/28165a4) 编写
