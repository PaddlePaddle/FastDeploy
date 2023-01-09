English | [简体中文](README_CN.md)
# UltraFace Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of UltraFace on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/facedet/ultraface/python/

# Download ultraface model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320.onnx
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# CPU inference
python infer.py --model version-RFB-320.onnx --image test_lite_face_detector_3.jpg --device cpu
# GPU inference
python infer.py --model version-RFB-320.onnx --image test_lite_face_detector_3.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model version-RFB-320.onnx --image test_lite_face_detector_3.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301821-0788483b-a72b-42b0-a566-b6430f184f6e.jpg">

## UltraFace Python Interface 

```python
fastdeploy.vision.facedet.UltraFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

UltraFace model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> UltraFace.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
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
> > Return `fastdeploy.vision.FaceDetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description.

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(list[int]): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [320, 240]

## Other Documents

- [UltraFace Model Description](..)
- [UltraFace C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
