English | [简体中文](README_CN.md)
# RobustVideoMatting C++ Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the RobustVideoMatting inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

This directory provides examples that `infer.cc` fast finishes the deployment of RobustVideoMatting on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy  Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download RobustVideoMatting model files, test images and videos
## Original ONNX model
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_fp32.onnx
## The ONNX model is specially processed for loading TRT
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_trt.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/video.mp4

# CPU inference
./infer_demo rvm_mobilenetv3_fp32.onnx matting_input.jpg matting_bgr.jpg 0
# GPU inference
./infer_demo rvm_mobilenetv3_fp32.onnx matting_input.jpg matting_bgr.jpg 1
# TRT inference
./infer_demo rvm_mobilenetv3_trt.onnx matting_input.jpg matting_bgr.jpg 2
```

The visualized result after running is as follows
<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## RobustVideoMatting C++ Interface 

```c++
fastdeploy::vision::matting::RobustVideoMatting(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

RobustVideoMatting model loading and initialization, among which model_file is the exported ONNX model format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict Function 

> ```c++
> RobustVideoMatting::Predict(cv::Mat* im, MattingResult* result)
> ```
>
> Model prediction interface. Input images and output matting results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Matting result. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of MattingResult


## Other Documents

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
