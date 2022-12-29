English | [简体中文](README.md)
# YOLOv5Face Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py`  fast finishes the deployment of YOLOv5Face on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/facedet/yolov5face/python/

# Download YOLOv5Face model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-face.onnx
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# CPU inference
python infer.py --model yolov5s-face.onnx --image test_lite_face_detector_3.jpg --device cpu
# GPU inference
python infer.py --model yolov5s-face.onnx --image test_lite_face_detector_3.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model yolov5s-face.onnx --image test_lite_face_detector_3.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301839-a29aefae-16c9-4196-bf9d-9c6cf694f02d.jpg">

## YOLOv5Face Python  Interface 

```python
fastdeploy.vision.facedet.YOLOv5Face(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv5Face model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> YOLOv5Face.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
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
> > Return `fastdeploy.vision.FaceDetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/)  for its description.

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(list[int]): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(list[float]): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding or not. `is_no_pad=True` represents no paddling. Default `is_no_pad=False`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=False`
> > * **stride**(int): Used with the `is_mini_pad` member variable. Default `stride=32`
> > * **landmarks_per_face**(int): Specify the number of keypoints in the face detected. Default 5

## 其它文档

- [YOLOv5Face 模型介绍](..)
- [YOLOv5Face C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)