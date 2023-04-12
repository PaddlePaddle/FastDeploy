English | [简体中文](README_CN.md)
# Smoke C++ Deployment Example

This directory provides an example of `infer.cc` to quickly complete the deployment of Smoke on CPU/GPU.

Before deployment, the following two steps need to be confirmed

- 1. The hardware and software environment meets the requirements, refer to [FastDeploy environment requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. According to the development environment, download the precompiled deployment library and samples code, refer to [FastDeploy prebuilt library](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

Taking CPU inference on Linux as an example, execute the following command in this directory to complete the compilation test. To support this model, you need to ensure FastDeploy version 1.0.6 or higher (x.x.x>=1.0.6)

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library, users can choose the appropriate version to use in the `FastDeploy precompiled library` mentioned above
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

wget https://bj.bcebos.com/fastdeploy/models/smoke.tar.gz
tar -xf smoke.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

# CPU
./infer_demo smoke smoke_test.png 0
# GPU
./infer_demo smoke smoke_test.png 1

```

The visual result after running is shown in the figure below

<img width="640" src="https://user-images.githubusercontent.com/30516196/230387825-53ac0a09-4137-4e49-9564-197cbc30ff08.png">

The above commands are only applicable to Linux or MacOS. For the usage of SDK under Windows, please refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## Smoke C++ interface

### Class Smoke

```c++
fastdeploy::vision::detection::Smoke(
        const string& model_file,
        const string& params_file,
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

Smoke model loading and initialization.

**parameter**

> * **model_file**(str): model file path
> * **params_file**(str): parameter file path
> * **config_file**(str): configuration file path
> * **runtime_option**(RuntimeOption): Backend reasoning configuration, the default is None, that is, the default configuration is used
> * **model_format**(ModelFormat): model format, the default is Paddle format

#### Predict function

> ```c++
> Smoke::Predict(cv::Mat* im, PerceptionResult* result)
> ```
>
> Model prediction interface, the input image directly outputs the detection result.
>
> **parameters**
>
> > * **im**: input image, note that it must be in HWC, BGR format
> > * **result**: Detection result, including the detection frame, the confidence of each frame, PerceptionResult description reference [visual model prediction results](../../../../../docs/api /vision_results/)


- [Model Introduction](../../)
- [Python deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
