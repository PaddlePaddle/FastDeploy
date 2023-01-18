English | [简体中文](README_CN.md)
# PP-PicoDet + PP-TinyPose Co-deployment (Pipeline)

## Model Description

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

Now FastDeploy supports the deployment of the following models

- [PP-PicoDet + PP-TinyPose Models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## Prepare PP-TinyPose Deployment Model

Export the PP-TinyPose and PP-PicoDet models. Please refer to [Model Export](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md) 

**Attention**: The exported inference model contains three files, including `model.pdmodel`、`model.pdiparams` and `infer_cfg.yml`.  FastDeploy will get the pre-processing information for inference from yaml files.

## Download Pre-trained Model

For developers' testing, part of the PP-PicoDet + PP-TinyPose（Pipeline）exported models are provided below. Developers can download and use them directly. 

| Application Scenario                          |  Model                                | Parameter File Size  |  AP(Service Data set) | AP(COCO Val Single/Multi-person) | Single/Multi-person Inference Time (FP32) | Single/Multi-person Inference Time（FP16) |
|:-------------------------------|:--------------------------------- |:----- |:----- | :----- | :----- | :----- |
| Single-person Model Configuration  |[PicoDet-S-Lcnet-Pedestrian-192x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_192x192_infer.tgz) + [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | 4.6MB + 5.3MB | 86.2% | 52.8% | 12.90ms | 9.61ms |
| Multi-person Model Configuration |[PicoDet-S-Lcnet-Pedestrian-320x320](https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz) + [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | 4.6M + 5.3MB | 85.7% | 49.9% | 47.63ms | 34.62ms |

**Note**
- The accuracy of the keypoint detection model is based on the detection frame obtained by the pedestrian detection model. 
- The flip operation is removed from the accuracy test with the detection confidence threshold of 0.5. 
- The speed test environment is qualcomm snapdragon 865 with 4-thread inference under arm8. 
- The Pipeline speed covers the preprocessing, inference, and post-processing of the model. 
- In the accuracy test, images with more than 6 people (excluding 6 people) were removed from the multi-person data for fair comparison.

For more information: refer to [PP-TinyPose official document](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
