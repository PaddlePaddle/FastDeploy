English | [简体中文](README_CN.md)
# PP-TinyPose Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides the `Multi-person keypoint detection in a single image` example that `pptinypose_infer.py` fast finishes the deployment of PP-TinyPose on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
>> **Attention**: single model currently only supports single-person keypoint detection in a single image. Therefore, the input image should contain one person only or should be cropped. For multi-person keypoint detection, refer to [PP-TinyPose Pipeline](../../det_keypoint_unite/python/README.md)

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/keypointdetection/tiny_pose/python

# Download PP-TinyPose model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/hrnet_demo.jpg

# CPU inference
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device cpu
# GPU inference
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device gpu
# TensorRT inference on GPU（Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device gpu --use_trt True
# KunlunXin XPU inference
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device kunlunxin
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196386764-dd51ad56-c410-4c54-9580-643f282f5a83.jpeg", width=359px, height=423px />
</div>

## PP-TinyPose Python Interface 

```python
fd.vision.keypointdetection.PPTinyPose(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PP-TinyPose model loading and initialization, among which model_file, params_file, and config_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md) for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict  function

> ```python
> PPTinyPose.predict(input_image)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.KeyPointDetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.

### Class Member Property
#### Post-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **use_dark**(bool): •	Whether to use DARK for post-processing. Refer to [Reference Paper](https://arxiv.org/abs/1910.06278)


## Other Documents

- [PP-TinyPose Model Description](..)
- [PP-TinyPose C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
