# ERNIE-3.0 模型部署

## 模型详细说明
- [PaddleNLP ERNIE-3.0模型说明](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0)

## 支持的模型列表

| 模型 |  结构  | 语言 |
| :---: | :--------: | :--------: |
| `ERNIE 3.0-Base`| 12-layers, 768-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Medium`| 6-layers, 768-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Mini`| 6-layers, 384-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Micro`| 4-layers, 384-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Nano `| 4-layers, 312-hidden, 12-heads | 中文 |

## 导出部署模型

在部署前，需要先将训练好的ERNIE模型导出成部署模型，导出步骤可参考文档[导出模型](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0).

## 下载基于AFQMC数据集的微调模型

为了方便开发者的测试，下面提供了在文本分类AFQMC数据集上微调的ERNIE-3.0-Medium模型，开发者可直接下载体验。

- [ERNIE 3.0-Medium AFQMC](https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz)

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
- [Serving部署](serving)
