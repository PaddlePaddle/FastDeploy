English | [简体中文](README_CN.md)
# FSANet C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of FSANet on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.2 or above (x.x.x>=1.0.2), or the nightly built version is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the official converted FSANet model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/fsanet-var.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/headpose_input.png
# CPU inference
./infer_demo --model fsanet-var.onnx --image headpose_input.png --device cpu
# GPU inference
./infer_demo --model fsanet-var.onnx --image headpose_input.png --device gpu
# TensorRT Inference on GPU
./infer_demo --model fsanet-var.onnx --image headpose_input.png --device gpu --backend trt
```

The visualized result after running is as follows

<div width="520">
<img width="500" height="514" float="left" src="https://user-images.githubusercontent.com/19977378/198279932-3eee424e-98a2-4249-bdeb-0f79127cbc9d.png">
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## FSANet C++ Interface 

### FSANet Class 

```c++
fastdeploy::vision::headpose::FSANet(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```
FSANet model loading and initialization, among which model_file is the exported ONNX model format.
**Parameter**
> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Only passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default
#### Predict Function
> ```c++
> FSANet::Predict(cv::Mat* im, HeadPoseResult* result)
> ```
>
> Model prediction interface. Input images and output head pose prediction results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Head pose prediction results. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of HeadPoseResult
### Class Member Variable
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results
> > * **size**(vector&lt;int&gt;): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [112, 112]

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
