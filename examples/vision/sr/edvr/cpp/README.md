English | [简体中文](README_CN.md)
# EDVR C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of EDVR on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the EDVR inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0)

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download EDVR model files and test videos
wget https://bj.bcebos.com/paddlehub/fastdeploy/EDVR_M_wo_tsa_SRx4.tar
tar -xvf EDVR_M_wo_tsa_SRx4.tar
wget https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4


# CPU inference
./infer_demo EDVR_M_wo_tsa_SRx4 vsr_src.mp4 0 5
# GPU inference
./infer_demo EDVR_M_wo_tsa_SRx4 vsr_src.mp4 1 5
# TensorRT Inference on GPU
./infer_demo EDVR_M_wo_tsa_SRx4 vsr_src.mp4 2 5
```

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## EDVR C++ Interface 

### EDVR Class

```c++
fastdeploy::vision::sr::EDVR(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

EDVR model loading and initialization, among which model_file is the exported Paddle model format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict Function

> ```c++
> EDVR::Predict(std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& results)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **imgs**: Input video frame sequence in HWC or BGR format
> > * **results**: Video SR results: video frame sequence after SR

- [Model Description](../../)
- [Python Deployment](../python)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
