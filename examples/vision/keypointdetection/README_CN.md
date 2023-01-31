[English](README.md) | 简体中文
# 关键点检测模型

FastDeploy目前支持两种关键点检测任务方式的部署

| 任务 | 说明 | 模型格式 | 示例 | 版本 |
| :---| :--- | :--- | :------- | :--- |
| 单人关键点检测 | 部署PP-TinyPose系列模型，输入图像仅包含单人 | Paddle | 参考[tinypose目录](./tiny_pose/) |  [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| 单人/多人关键点检测 | 部署PicoDet + PP-TinyPose的模型串联任务，输入图像先通过检测模型，得到独立的人像子图后，再经过PP-TinyPose模型检测关键点 | Paddle | 参考[det_keypoint_unite目录](./det_keypoint_unite/) |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |

# 预训练模型准备
本文档提供了如下预训练模型，开发者可直接下载使用
| 模型 | 说明 | 模型格式 | 版本 |
| :--- | :--- | :------- | :--- |
| [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | 单人关键点检测模型 | Paddle | [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz) | 单人关键点检测模型 | Paddle | [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PicoDet-S-Lcnet-Pedestrian-192x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_192x192_infer.tgz) + [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | 单人关键点检测串联配置 | Paddle |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PicoDet-S-Lcnet-Pedestrian-320x320](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz) + [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | 多人关键点检测串联配置 | Paddle |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
