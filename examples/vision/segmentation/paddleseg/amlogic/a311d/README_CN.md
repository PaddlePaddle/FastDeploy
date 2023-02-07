[English](README.md) | 简体中文
# 在晶晨A311D上使用FastDeploy部署PaddleSeg模型
晶晨A311D是一款先进的AI应用处理器。目前，FastDeploy支持在A311D上基于Paddle-Lite部署PaddleSeg相关模型

## 晶晨A311D支持的PaddleSeg模型
由于晶晨A311D的NPU仅支持INT8量化模型的部署，因此所支持的量化模型如下：
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型，开发者可直接下载使用。

| 模型                              | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [PP-LiteSeg-T(STDC1)-cityscapes-without-argmax](https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz)| 31MB  | 1024x512 | 77.04% | 77.73% | 77.46% |
>> **注意**: FastDeploy模型量化的方法及一键自动化压缩工具可以参考[模型量化](../../../quantize/README.md)

## 详细部署文档

目前，A311D上只支持C++的部署。

- [C++部署](cpp)
