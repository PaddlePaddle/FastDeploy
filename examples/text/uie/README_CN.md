[English](README.md) | 简体中文

# 通用信息抽取 UIE模型部署

## 模型版本说明

- [PaddleNLP 通用信息抽取UIE](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/uie)

## 支持的模型列表

| 模型 |  结构  | 语言 |
| :---: | :--------: | :--------: |
| `uie-base`| 12-layers, 768-hidden, 12-heads | 中文 |
| `uie-medium`| 6-layers, 768-hidden, 12-heads | 中文 |
| `uie-mini`| 6-layers, 384-hidden, 12-heads | 中文 |
| `uie-micro`| 4-layers, 384-hidden, 12-heads | 中文 |
| `uie-nano`| 4-layers, 312-hidden, 12-heads | 中文 |


## 导出部署模型

在部署前，需要先将UIE模型导出成部署模型，导出步骤参考文档[导出模型](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/uie#47-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)

## 下载预训练模型

为了方便开发者的测试，下面提供了UIE导出的各模型，开发者可直接下载使用。

其中精度指标F1来源于PaddleNLP中对各模型在互联网自建数据集进行0-shot的实验，详情各参考PaddleNLP中的说明。

| 模型                                                               | 参数大小    | F1 值|
|:---------------------------------------------------------------- |:----- |:----- |
|[uie-base](https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz)| 416M | 78.33	|
|[uie-medium](https://bj.bcebos.com/fastdeploy/models/uie/uie-medium.tgz)| 265M | 78.32 |
|[uie-mini](https://bj.bcebos.com/fastdeploy/models/uie/uie-mini.tgz)| 95M | 72.09 |
|[uie-micro](https://bj.bcebos.com/fastdeploy/models/uie/uie-micro.tgz)| 83M | 66.00 |
|[uie-nano](https://bj.bcebos.com/fastdeploy/models/uie/uie-nano.tgz)| 64M | 62.86 |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
- [服务化部署](serving)
