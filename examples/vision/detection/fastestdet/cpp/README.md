English | [简体中文](README_CN.md)
# FastestDet C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of FastestDet on CPU/GPU and GPU accelerated by TensorRT. 
Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. 

```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-1.0.3.tgz
tar xvf fastdeploy-linux-x64-1.0.3.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-1.0.3
make -j

# Download the official converted FastestDet model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/FastestDet.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# CPU inference
./infer_demo FastestDet.onnx 000000014439.jpg 0
# GPU inference
./infer_demo FastestDet.onnx 000000014439.jpg 1
# TensorRT inference on GPU
./infer_demo FastestDet.onnx 000000014439.jpg 2
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/44280887/206176291-61eb118b-391b-4431-b79e-a393b9452138.jpg">

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## FastestDet C++ Interface 

### FastestDet Class

```c++
fastdeploy::vision::detection::FastestDet(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

FastestDet model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Only passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict Function

> ```c++
> FastestDet::Predict(cv::Mat* im, DetectionResult* result,
>                 float conf_threshold = 0.65,
>                 float nms_iou_threshold = 0.45)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for DetectionResult
> > * **conf_threshold**: Filtering threshold of detection box confidence
> > * **nms_iou_threshold**: iou threshold during NMS processing

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(vector&lt;int&gt;): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [352, 352]

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
