English | [简体中文](README_CN.md)
# Keypoint Detection Model

Currently FastDeploy supports the deployment of two key point detection methods.

| Task | Description | Model Format | Example | Version |
| :---| :--- | :--- | :------- | :--- |
| Single person key point detection | Deploy PP-TinyPose series models with input images containing only a single person | Paddle | Please refer to [tinypose](./tiny_pose/) |  [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| Single/multi person key point detection | Deploy the model tandem task of PicoDet + PP-TinyPose, where the input image first passes through the detection model to get an independent portrait sub-image, and then passes through the PP-TinyPose model to detect key points | Paddle | Please refer to [det_keypoint_unite](./det_keypoint_unite/) |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |

# Prepare Pre-trained Model
Here we provides the following pre-trained models, you can download and use them directly.
| Model | Description | Model Format | Version |
| :--- | :--- | :------- | :--- |
| [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | Single person key point detection | Paddle | [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz) | Single person key point detection | Paddle | [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PicoDet-S-Lcnet-Pedestrian-192x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_192x192_infer.tgz) + [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | Single person key point detection tandem configuration | Paddle |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PicoDet-S-Lcnet-Pedestrian-320x320](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz) + [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | Multi person key point detection tandem configuration | Paddle |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
