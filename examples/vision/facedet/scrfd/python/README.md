English | [简体中文](README_CN.md)
# SCRFD Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of SCRFD on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/facedet/scrfd/python/

# Download SCRFD model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape640x640.onnx
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# CPU inference
python infer.py --model scrfd_500m_bnkps_shape640x640.onnx --image test_lite_face_detector_3.jpg --device cpu
# GPU inference
python infer.py --model scrfd_500m_bnkps_shape640x640.onnx --image test_lite_face_detector_3.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model scrfd_500m_bnkps_shape640x640.onnx --image test_lite_face_detector_3.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301789-1981d065-208f-4a6b-857c-9a0f9a63e0b1.jpg">

## SCRFD Python Interface 

```python
fastdeploy.vision.facedet.SCRFD(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

SCRFD model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> SCRFD.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format
> > * **conf_threshold**(float): Filtering threshold of detection box confidence
> > * **nms_iou_threshold**(float): iou threshold during NMS processing

> **Return**
>
> > Return `fastdeploy.vision.FaceDetectionResult`  structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/)  for its description.

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(list[int]): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(list[float]): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding or not. `is_no_pad=True` represents no paddling. Default `is_no_pad=False`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=False`
> > * **stride**(int): Used with the `stris_mini_padide` member variable. Default`stride=32`
> > * **downsample_strides**(list[int]): This parameter is used to change the down-sampling multiple of the feature map that generates anchor, containing three integer elements that represent the default down-sampling multiple for generating anchor. Default [8, 16, 32]
> > * **landmarks_per_face**(int): Modify the number of face keypoints if we use an output with face keypoints. Default `landmarks_per_face=5`
> > * **use_kps**(bool): Whether to use keypoints or not. If the ONNX file has no keypoint output, set `use_kps=False` and `landmarks_per_face=0`. Default `use_kps=True`
> > * **num_anchors**(int): Set the number predicted by each anchor. The parameters of the trained model need modification accordingly. Default `num_anchors=2`


## Other Documents

- [SCRFD Model Description](..)
- [SCRFD C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
