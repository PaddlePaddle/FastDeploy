[English](README.md) | 简体中文
# 在瑞芯微 RV1126 上使用 FastDeploy 部署 PaddleSeg 模型
瑞芯微 RV1126 是一款编解码芯片，专门面相人工智能的机器视觉领域。目前，FastDeploy 支持在 RV1126 上基于 Paddle-Lite 部署 PaddleSeg 相关模型

## 瑞芯微 RV1126 支持的PaddleSeg模型
由于瑞芯微 RV1126 的 NPU 仅支持 INT8 量化模型的部署，因此所支持的量化模型如下：
- [PP-LiteSeg 系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)

为了方便开发者的测试，下面提供了 PaddleSeg 导出的部分模型，开发者可直接下载使用。

| 模型                              | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [PP-LiteSeg-T(STDC1)-cityscapes-without-argmax](https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz)| 31MB  | 1024x512 | 77.04% | 77.73% | 77.46% |
>> **注意**: FastDeploy 模型量化的方法及一键自动化压缩工具可以参考[模型量化](../../../quantize/README.md)

## 详细部署文档

目前，瑞芯微 RV1126 上只支持C++的部署。

- [C++部署](cpp)
