English | [简体中文](README_CN.md)
# YOLOX Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy  Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of YOLOX on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/yolox/python/

# Download YOLOX model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
python infer.py --model yolox_s.onnx --image 000000014439.jpg --device cpu
# GPU inference
python infer.py --model yolox_s.onnx --image 000000014439.jpg --device gpu
# TensorRT inference on GPU (TensorRT in SDK. No need Separate installation)
python infer.py --model yolox_s.onnx --image 000000014439.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301746-04595d76-454a-4f07-8c7d-6f41418f8ae3.jpg">

## YOLOX Python Interface

```python
fastdeploy.vision.detection.YOLOX(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOX model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> YOLOX.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
>
> Model prediction interface. Input images and output results
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format
> > * **conf_threshold**(float): Filtering threshold of detection box confidence
> > * **nms_iou_threshold**(float): iou threshold during NMS processing

> **Return**
>
> > Return `fastdeploy.vision.DetectionResult` structure, refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> >* **size**(list[int]): This parameter changes the size of the resize during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> >* **padding_value**(list[float]): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default [114, 114, 114]
> >* **is_decode_exported**(bool): The default value is `is_decode_exported=False`. The official default export does not have the decoded part. If you export the model with the decoded part, please set this parameter to true



## Other Documents

- [YOLOX Model Description](..)
- [YOLOX C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
