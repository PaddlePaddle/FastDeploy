English | [简体中文](README_CN.md)
# YOLOv7End2EndORT C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of YOLOv7End2EndORT on CPU/GPU accelerated by TensorRT. 

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the official converted yolov7 model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-end2end-ort-nms.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# CPU inference
./infer_demo yolov7-end2end-ort-nms.onnx 000000014439.jpg 0
# GPU inference
./infer_demo yolov7-end2end-ort-nms.onnx 000000014439.jpg 1
# TensorRT + GPU deployment (Not supported yet. Back to ORT + GPU)
./infer_demo yolov7-end2end-ort-nms.onnx 000000014439.jpg 2
```

The visualized result after running is as follows

<div align='center'>
  <img width="639" alt="image" src="https://user-images.githubusercontent.com/31974251/186369053-1b578d61-ca70-4755-9671-c9fccf6314a0.png">
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

Attention: YOLOv7End2EndORT is designed for the inference of End2End models with [ORT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L87) among the YOLOv7 exported models. For models without nms, use YOLOv7 class for inference. For End2End models with [TRT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L111), use YOLOv7End2EndTRT for inference.

## YOLOv7End2EndORT C++ Interface 

### YOLOv7End2EndORT Class

```c++
fastdeploy::vision::detection::YOLOv7End2EndORT(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv7End2EndORT model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Merely passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict function

> ```c++
> YOLOv7End2EndORT::Predict(cv::Mat* im, DetectionResult* result,
>                           float conf_threshold = 0.25)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for DetectionResult 
> > * **conf_threshold**: Filtering threshold of detection box confidence. But considering that YOLOv7 End2End models have a score threshold specified during ONNX export, this parameter will be effective when being greater than the specified one.

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(vector&lt;int&gt;): This parameter changes resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(vector&lt;float&gt;): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding. `is_no_pad=ture` represents no paddling. Default`is_no_pad=false`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=false`
> > * **stride**(int): Used with the `stris_mini_pad` member variable. Default `stride=32`

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
