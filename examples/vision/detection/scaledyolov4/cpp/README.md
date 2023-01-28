English | [简体中文](README_CN.md)
# ScaledYOLOv4 C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of ScaledYOLOv4 on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the official converted ScaledYOLOv4 model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p5.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# CPU inference
./infer_demo scaled_yolov4-p5.onnx 000000014439.jpg 0
# GPU inference
./infer_demo scaled_yolov4-p5.onnx 000000014439.jpg 1
# TensorRT inference on GPU
./infer_demo scaled_yolov4-p5.onnx 000000014439.jpg 2
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301908-7027cf41-af51-4485-bd32-87aca0e77336.jpg">

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to: 
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## ScaledYOLOv4 C++ Interface 

### ScaledYOLOv4 Class

```c++
fastdeploy::vision::detection::ScaledYOLOv4(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

ScaledYOLOv4 model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Merely passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict Parameter

> ```c++
> ScaledYOLOv4::Predict(cv::Mat* im, DetectionResult* result,
>                 float conf_threshold = 0.25,
>                 float nms_iou_threshold = 0.5)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for DetectionResult
> > * **conf_threshold**:  Filtering threshold of detection box confidence
> > * **nms_iou_threshold**: iou threshold during NMS processing

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(vector&lt;int&gt;): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(vector&lt;float&gt;): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding or not. `is_no_pad=ture` represents no paddling. Default `is_no_pad=false`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=false`
> > * **stride**(int): Used with the `stris_mini_pad` member variable. Default `stride=32`

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
