English | [简体中文](README_CN.md)
# PP-Tracking C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of PP-Tracking on CPU/GPU and GPU accelerated by TensorRT.
Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the PP-Tracking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the`FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download PP-Tracking model files and test videos
wget https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
tar -xvf fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4


# CPU inference
./infer_demo fairmot_hrnetv2_w18_dlafpn_30e_576x320 person.mp4 0
# GPU inference
./infer_demo fairmot_hrnetv2_w18_dlafpn_30e_576x320 person.mp4 1
# TensorRT Inference on GPU
./infer_demo fairmot_hrnetv2_w18_dlafpn_30e_576x320 person.mp4 2
```

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## PP-Tracking C++ Interface 

### PPTracking Class 

```c++
fastdeploy::vision::tracking::PPTracking(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PP-Tracking model loading and initialization, among which model_file is the exported Paddle model format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict Function

> ```c++
> PPTracking::Predict(cv::Mat* im, MOTResult* result)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box, tracking id, confidence of each box, and object class id. Refer to [visual model prediction results](../../../../../docs/api/vision_results/) for the description of MOTResult


- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
