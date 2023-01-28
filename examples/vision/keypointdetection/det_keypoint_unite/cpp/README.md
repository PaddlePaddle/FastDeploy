English | [简体中文](README_CN.md)
# PP-PicoDet + PP-TinyPose (Pipeline) C++ Deployment Example

This directory provides the `Multi-person keypoint detection in a single image` example that `det_keypoint_unite_infer.cc` fast finishes the deployment of multi-person detection model PP-PicoDet + PP-TinyPose on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)


Taking the inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy  Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download PP-TinyPose+PP-PicoDet model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
tar -xvf PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/000000018491.jpg

# CPU inference
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 0
# GPU inference
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 1
# TensorRT inference on GPU
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 2
# kunlunxin XPU inference
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 3
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196393343-eeb6b68f-0bc6-4927-871f-5ac610da7293.jpeg", width=359px, height=423px />
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## PP-TinyPose C++ Interface 

### PP-TinyPose Class

```c++
fastdeploy::pipeline::PPTinyPose(
        fastdeploy::vision::detection::PPYOLOE* det_model,
        fastdeploy::vision::keypointdetection::PPTinyPose* pptinypose_model)
```

PPTinyPose Pipeline model loading and initialization.

**Parameter**

> * **model_det_modelfile**(fastdeploy::vision::detection): Initialized detection model. Refer to [PP-TinyPose](../../tiny_pose/README.md)
> * **pptinypose_model**(fastdeploy::vision::keypointdetection): Initialized detection model [Detection](../../../detection/paddledetection/README.md). Currently only PaddleDetection series is available.

#### Predict Function

> ```c++
> PPTinyPose::Predict(cv::Mat* im, KeyPointDetectionResult* result)
> ```
>
> Model prediction interface. Input images and output keypoint detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Keypoint detection results, including coordinates and the corresponding probability value. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of KeyPointDetectionResult

### Class Member Property
#### Post-processing Parameter
> > * **detection_model_score_threshold**(bool): Score threshold of the Detectin model for filtering detection boxes before entering the PP-TinyPose model

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
