[English](README.md) | 简体中文
# PP-TinyPose 模型部署

## 模型版本说明

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

目前FastDeploy支持如下模型的部署 

- [PP-TinyPose系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## 准备PP-TinyPose部署模型

PP-TinyPose模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md)  

**注意**:PP-TinyPose导出的模型包含`model.pdmodel`、`model.pdiparams`和`infer_cfg.yml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息。


## 下载预训练模型

为了方便开发者的测试，下面提供了PP-TinyPose导出的部分模型，开发者可直接下载使用。

| 模型                                                               | 参数文件大小 |输入Shape |  AP(业务数据集) | AP(COCO Val) | FLOPS | 单人推理耗时 (FP32) | 单人推理耗时（FP16) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- | :----- | :----- |
| [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | 5.3MB | 128x96 | 84.3% | 58.4% | 81.56 M | 4.57ms | 3.27ms |
| [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | 5.3M  | 256x96 | 91.0% | 68.3% | 326.24M | 14.07ms | 8.33ms |

**说明**
- 关键点检测模型使用`COCO train2017`和`AI Challenger trainset`作为训练集。使用`COCO person keypoints val2017`作为测试集。
- 关键点检测模型的精度指标所依赖的检测框为ground truth标注得到。
- 推理速度测试环境为 Qualcomm Snapdragon 865，采用arm8下4线程推理得到。

更多信息请参考：[PP-TinyPose 官方文档](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
