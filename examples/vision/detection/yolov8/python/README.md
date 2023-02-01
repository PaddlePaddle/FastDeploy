English | [简体中文](README_CN.md)
# YOLOv8 Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl. Refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides the example that `infer.py` fast finishes the deployment of YOLOv8 on CPU/GPU and GPU through TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/yolov8/python/

# Download yolov8 model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov8.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
python infer.py --model yolov8.onnx --image 000000014439.jpg --device cpu
# GPU inference
python infer.py --model yolov8.onnx --image 000000014439.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model yolov8.onnx --image 000000014439.jpg --device gpu --use_trt True
```

The visualized result is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184309358-d803347a-8981-44b6-b589-4608021ad0f4.jpg">

## YOLOv8 Python Interface

```python
fastdeploy.vision.detection.YOLOv8(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv8 model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> YOLOv8.predict(image_data)
> ```
>
> Model prediction interface. Input images and output detection results
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return the `fastdeploy.vision.DetectionResult`structure, refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description

### Class Member Property
#### Pre-processing Parameter
Users can modify the following preprocessing parameters based on actual needs to change the final inference and deployment results

> > * **size**(list[int]): This parameter changes the resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(list[float]): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding. `is_no_pad=True` represents no paddling. Default `is_no_pad=False`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=False`
> > * **stride**(int): Used with the `stris_mini_padide` member variable. Default `stride=32`

## Other Documents

- [YOLOv8 Model Description](..)
- [YOLOv8 C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
