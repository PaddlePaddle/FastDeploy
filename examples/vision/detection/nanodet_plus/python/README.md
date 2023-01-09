English | [简体中文](README_CN.md)
# NanoDetPlus Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of NanoDetPlus on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/nanodet_plus/python/

# Download NanoDetPlus model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
python infer.py --model nanodet-plus-m_320.onnx --image 000000014439.jpg --device cpu
# GPU inference
python infer.py --model nanodet-plus-m_320.onnx --image 000000014439.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model nanodet-plus-m_320.onnx --image 000000014439.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301689-87ee5205-2eff-4204-b615-24c400f01323.jpg">

## NanoDetPlus Python Interface 

```python
fastdeploy.vision.detection.NanoDetPlus(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

NanoDetPlus model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> NanoDetPlus.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
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

> > * **size**(list[int]): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [320, 320]
> > * **padding_value**(list[float]): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [0, 0, 0]
> > * **keep_ratio**(bool): Whether to keep the aspect ratio unchanged during resize. Default false
> > * **reg_max**(int): The reg_max parameter in GFL regression. Default 7.
> > * **downsample_strides**(list[int]): This parameter is used to change the down-sampling multiple of the feature map that generates anchor, containing four integer elements that represent the default down-sampling multiple for generating anchor. Default [8, 16, 32, 64]



## Other Documents

- [NanoDetPlus Model Description](..)
- [NanoDetPlus C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
