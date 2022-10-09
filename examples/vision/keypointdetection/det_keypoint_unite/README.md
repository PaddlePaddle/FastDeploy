# PP-PicoDet + PP-TinyPose 模型(Pipeline)部署

## 模型版本说明

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

目前FastDeploy支持如下模型的部署 

- [PP-PicoDet + PP-TinyPose系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## 准备PP-TinyPose部署模型

PP-TinyPose以及PP-PicoDet模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md)  

**注意**:导出的推理模型包含`model.pdmodel`、`model.pdiparams`和`infer_cfg.yml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息。


## 下载预训练模型

为了方便开发者的测试，下面提供了PP-PicoDet + PP-TinyPose（Pipeline）导出的部分模型，开发者可直接下载使用。

| 应用场景                         |  模型                                 | 参数文件大小 |  AP(业务数据集) | AP(COCO Val 单人/多人) | 单人/多人推理耗时 (FP32) | 单人/多人推理耗时（FP16) |
|:-------------------------------|:--------------------------------- |:----- |:----- | :----- | :----- | :----- |
| 单人模型配置 |[PicoDet-S-Lcnet-Pedestrian-192x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_192x192_infer.tgz) + [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | 4.6MB + 5.3MB | 86.2% | 52.8% | 12.90ms | 9.61ms |
| 多人模型配置 |[PicoDet-S-Lcnet-Pedestrian-320x320](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz) + [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | 4.6M + 5.3MB | 85.7% | 49.9% | 47.63ms | 34.62ms |

**说明**
- 关键点检测模型的精度指标是基于对应行人检测模型检测得到的检测框。
- 精度测试中去除了flip操作，且检测置信度阈值要求0.5。
- 速度测试环境为qualcomm snapdragon 865，采用arm8下4线程推理。
- Pipeline速度包含模型的预处理、推理及后处理部分。
- 精度测试中，为了公平比较，多人数据去除了6人以上（不含6人）的图像。

更多信息请参考：[PP-TinyPose 官方文档](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
