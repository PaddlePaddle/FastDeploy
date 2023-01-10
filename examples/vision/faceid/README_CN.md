[English](README.md) | 简体中文
# 人脸识别模型


## 模型支持列表

FastDeploy目前支持如下人脸识别模型部署

| 模型                                     | 说明             | 模型格式       | 版本                                                                            |
|:---------------------------------------|:---------------|:-----------|:------------------------------------------------------------------------------|
| [deepinsight/ArcFace](./insightface)   | ArcFace 系列模型   | ONNX       | [CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) |
| [deepinsight/CosFace](./insightface)   | CosFace 系列模型   | ONNX       | [CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) |
| [deepinsight/PartialFC](./insightface) | PartialFC 系列模型 | ONNX       | [CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) |
| [deepinsight/VPL](./insightface)       | VPL 系列模型       | ONNX       | [CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) |
| [paddleclas/AdaFace](./adaface)        | AdaFace 系列模型   | PADDLE     | [CommitID:babb9a5](https://github.com/PaddlePaddle/PaddleClas/tree/v2.4.0)    |

## 模型demo简介

ArcFace,CosFace,PartialFC,VPL同属于deepinsight系列，因此demo使用ONNX作为推理框架。AdaFace则采用PaddleInference作为推理框架。
