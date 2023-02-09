[English](README.md) | 简体中文
# PaddleSeg在瑞芯微 RV1126上通过FastDeploy部署模型
瑞芯微 RV1126 是一款编解码芯片，专门面相人工智能的机器视觉领域。PaddleSeg支持通过FastDeploy在RV1126上基于Paddle-Lite部署相关Segmentation模型

## 瑞芯微 RV1126支持的PaddleSeg模型

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
>> **注意**：支持PaddleSeg高于2.6版本的Segmentation模型

目前瑞芯微 RV1126 的 NPU 支持的量化模型如下：
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)

## 预导出的量化推理模型
为了方便开发者的测试，下面提供了PaddleSeg导出的部分量化后的推理模型，开发者可直接下载使用。

| 模型                              | 参数文件大小    |输入Shape |  mIoU | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- |
| [PP-LiteSeg-T(STDC1)-cityscapes-without-argmax](https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz)| 31MB  | 1024x512 | 77.04% | 77.73% | 77.46% |
**注意**
- PaddleSeg量化模型包含`model.pdmodel`、`model.pdiparams`、`deploy.yaml`和`subgraph.txt`四个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息，subgraph.txt是为了异构计算而存储的配置文件
- 若以上列表中无满足要求的模型，可参考下方教程自行导出适配A311D的模型

## PaddleSeg动态图模型导出为RV1126支持的INT8模型
模型导出分为以下两步
1. PaddleSeg训练的动态图模型导出为推理静态图模型，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)
瑞芯微RV1126仅支持INT8
2. 将推理模型量化压缩为INT8模型，FastDeploy模型量化的方法及一键自动化压缩工具可以参考[模型量化](../../../quantize/README.md)

## 详细部署文档

目前，瑞芯微 RV1126 上只支持C++的部署。

- [C++部署](cpp)
