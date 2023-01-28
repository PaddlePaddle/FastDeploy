English | [简体中文](README_CN.md)
# YOLOv5Lite Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of YOLOv5Lite on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/yolov5lite/python/

# Download YOLOv5Lite model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-g-sim-640.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
python infer.py --model v5Lite-g-sim-640.onnx --image 000000014439.jpg --device cpu
# GPU inference
python infer.py --model v5Lite-g-sim-640.onnx --image 000000014439.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model v5Lite-g-sim-640.onnx --image 000000014439.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301943-263c8153-a52a-4533-a7c1-ee86d05d314b.jpg">

## YOLOv5Lite Python Interface 

```python
fastdeploy.vision.detection.YOLOv5Lite(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv5Lite model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> YOLOv5Lite.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
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
> > Return `fastdeploy.vision.DetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description.

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(list[int]): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(list[float]): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding. `is_no_pad=True`  represents no paddling. Default `is_no_pad=False`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=False`
> > * **stride**(int): Used with the `stris_mini_padide` member variable. Default `stride=32`



## Other Documents

- [YOLOv5Lite Model Description](..)
- [YOLOv5Lite C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
