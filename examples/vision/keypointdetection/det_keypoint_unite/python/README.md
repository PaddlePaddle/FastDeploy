English | [简体中文](README_CN.md)
# PP-PicoDet + PP-TinyPose (Pipeline) Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy  Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides the `Multi-person keypoint detection in a single image` example that `det_keypoint_unite_infer.py` fast finishes the deployment of multi-person detection model PP-PicoDet + PP-TinyPose on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
>> **Attention**: For standalone deployment of PP-TinyPose single model, refer to [PP-TinyPose Single Model](../../tiny_pose//python/README.md)

```bash
# Download the deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/keypointdetection/det_keypoint_unite/python

# Download PP-TinyPose model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
tar -xvf PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/000000018491.jpg
# CPU inference
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device cpu
# GPU inference
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device gpu
# TensorRT inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device gpu --use_trt True
# kunlunxin XPU inference
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device kunlunxin
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196393343-eeb6b68f-0bc6-4927-871f-5ac610da7293.jpeg", width=640px, height=427px />
</div>

## PPTinyPosePipeline Python Interface 

```python
fd.pipeline.PPTinyPose(det_model=None, pptinypose_model=None)
```

PPTinyPosePipeline model loading and initialization, among which the det_model is the detection model initialized by `fd.vision.detection.PicoDet`[Refer to Detection Document](../../../detection/paddledetection/python/) and pptinypose_model is the detection model initialized by `fd.vision.keypointdetection.PPTinyPose`[Refer to PP-TinyPose Document](../../tiny_pose/python/)

**Parameter**

> * **det_model**(str): Initialized detection model
> * **pptinypose_model**(str): Initialized PP-TinyPose model

### predict function

> ```python
> PPTinyPosePipeline.predict(input_image)
> ```
>
> Model prediction interface. Input images and output keypoint detection results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.KeyPointDetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.

### Class Member Property
#### Post-processing Parameter
> > * **detection_model_score_threshold**(bool):
Score threshold of the Detectin model for filtering detection boxes before entering the PP-TinyPose model

## Other Documents

- [Pipeline Model Description](..)
- [Pipeline C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
