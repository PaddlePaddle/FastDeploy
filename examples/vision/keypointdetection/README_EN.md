English | [简体中文](README.md)
# Keypoint Detection Model

Now FastDeploy supports the deployment of two kinds of keypoint detection tasks.

| Task | Description | Model Format | Example | Version |
| :---| :--- | :--- | :------- | :--- |
| Single-person keypoint detection | Deploy PP-TinyPose models with input image containing one single person | Paddle | Refer to [tinypose directory](./tiny_pose/) |  [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| Single-person/ multi-person keypoint detection | Deploy PicoDet + PP-TinyPose model concatenation task. After passing through the detection model to get an independent portrait sub-image, the input image will pass through the PP-TinyPose model to detect keypoints | Paddle | Refer to [det_keypoint_unite directory](./det_keypoint_unite/) |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |

# Prepare Pre-trained Model
The following pre-trained models are provided. Developers can download and use them directly.
| Model | Description | Model Format | Version |
| :--- | :--- | :------- | :--- |
| [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | Single-person keypoint detection model  | Paddle | [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz) | Single-person keypoint detection model | Paddle | [Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PicoDet-S-Lcnet-Pedestrian-192x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_192x192_infer.tgz) + [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | Single-person keypoint detection concatenated configuration | Paddle |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
| [PicoDet-S-Lcnet-Pedestrian-320x320](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz) + [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | Multi-person keypoint detection concatenated configuration | Paddle |[Release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) |
