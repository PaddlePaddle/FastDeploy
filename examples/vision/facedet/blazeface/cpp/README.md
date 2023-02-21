English | [简体中文](README_CN.md)
# BlazeFace C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of BlazeFace on CPU/GPU。

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz # x.x.x >= 1.0.4
tar xvf fastdeploy-linux-x64-x.x.x.tgz # x.x.x >= 1.0.4
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x # x.x.x >= 1.0.4
make -j

#Download the official converted YOLOv7Face model files and test images
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/blzeface-1000e.tgz

#Use blazeface-1000e model
# CPU inference
./infer_demo blazeface-1000e/ test_lite_face_detector_3.jpg 0
# GPU Inference
./infer_demo blazeface-1000e/ test_lite_face_detector_3.jpg 1
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/49013063/206170111-843febb6-67d6-4c46-a121-d87d003bba21.jpg">

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## BlazeFace C++ Interface

### BlazeFace Class

```c++
fastdeploy::vision::facedet::BlazeFace(
        const string& model_file,
        const string& params_file = "",
        const string& config_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

BlazeFace model loading and initialization, among which model_file is the exported PADDLE model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Only passing an empty string when the model is in PADDLE format
> * **config_file**(str): Config file path. Only passing an empty string when the model is in PADDLE format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. PADDLE format by default

#### Predict Function

> ```c++
> BlazeFace::Predict(cv::Mat& im, FaceDetectionResult* result)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Result](../../../../../docs/api/vision_results/) for FaceDetectionResult

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
