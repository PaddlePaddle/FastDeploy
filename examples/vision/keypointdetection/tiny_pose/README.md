English | [简体中文](README_CN.md)
# PP-TinyPose Model Deployment

## Model Description

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

Now FastDeploy supports the deployment of the following models 

- [PP-TinyPose models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## Prepare PP-TinyPose Deployment Model

Export the PP-TinyPose model. Please refer to [Model Export](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md)  

**Attention**: The exported PP-TinyPose model contains three files, including `model.pdmodel`、`model.pdiparams` and `infer_cfg.yml`. FastDeploy will get the pre-processing information for inference from yaml files.


## Download Pre-trained Model

For developers' testing, part of the PP-TinyPose exported models are provided below. Developers can download and use them directly. 

| Model                                                               | Parameter File Size  | Input Shape |  AP(Service Data set) | AP(COCO Val) | FLOPS | Single/Multi-person Inference Time (FP32) | Single/Multi-person Inference Time（FP16) |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- | :----- | :----- | :----- |
| [PP-TinyPose-128x96](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_128x96_infer.tgz) | 5.3MB | 128x96 | 84.3% | 58.4% | 81.56 M | 4.57ms | 3.27ms |
| [PP-TinyPose-256x192](https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz)  | 5.3M  | 256x96 | 91.0% | 68.3% | 326.24M | 14.07ms | 8.33ms |

**Note**
- The keypoint detection model uses `COCO train2017` and `AI Challenger trainset` as the training sets and `COCO person keypoints val2017` as the test set. 
- The detection frame, through which we get the accuracy of the keypoint detection model, is obtained from the ground truth annotation. 
- The speed test environment is Qualcomm Snapdragon 865 with 4-thread inference under arm8. 


For more information: refer to [PP-TinyPose official document](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
