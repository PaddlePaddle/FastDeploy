English | [简体中文](README_CN.md)
# YOLOv8 C++ Deployment Example

This directory provides the example that `infer.cc` fast finishes the deployment of YOLOv8 on CPU/GPU and GPU through TensorRT.

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code based on your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, FastDeploy version 1.0.3 or above (x.x.x>=1.0.3) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 1. Download the official converted YOLOv8 ONNX model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov8s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
./infer_demo yolov8s.onnx 000000014439.jpg 0
# GPU inference
./infer_demo yolov8s.onnx 000000014439.jpg 1
# TensorRT inference on GPU 
./infer_demo yolov8s.onnx 000000014439.jpg 2
```
The visualized result is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184309358-d803347a-8981-44b6-b589-4608021ad0f4.jpg">

he above command works for Linux or MacOS. For SDK in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

If you use Huawei Ascend NPU deployment, refer to the following document  to initialize the deployment environment:
- [How to use Huawei Ascend NPU deployment](../../../../../docs/cn/faq/use_sdk_on_ascend.md)

## YOLOv8 C++ Interface

### YOLOv8

```c++
fastdeploy::vision::detection::YOLOv8(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv8 model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Merely passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict function

> ```c++
> YOLOv8::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> Model prediction interface. Input images and output detection results
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for DetectionResult.

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following preprocessing parameters based on actual needs to change the final inference and deployment results

> > * **size**(vector&lt;int&gt;): This parameter changes the resize used during preprocessing, containing two integer elements for [width, height] with default value [640, 640]
> > * **padding_value**(vector&lt;float&gt;): This parameter is used to change the padding value of images during resize, containing three floating-point elements that represent the value of three channels. Default value [114, 114, 114]
> > * **is_no_pad**(bool): Specify whether to resize the image through padding. `is_no_pad=ture` represents no paddling. Default `is_no_pad=false`
> > * **is_mini_pad**(bool): This parameter sets the width and height of the image after resize to the value nearest to the `size` member variable and to the point where the padded pixel size is divisible by the `stride` member variable. Default `is_mini_pad=false`
> > * **stride**(int): Used with the `stris_mini_pad` member variable. Default `stride=32`

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
