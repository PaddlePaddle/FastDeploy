English | [简体中文](README_CN.md)
# AnimeGAN C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of AnimeGAN on CPU/GPU.

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy  Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the AnimeGAN inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.2 or above (x.x.x>=1.0.2) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the prepared model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/style_transfer_testimg.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/animegan_v1_hayao_60_v1.0.0.tgz
tar xvfz animegan_v1_hayao_60_v1.0.0.tgz

# CPU inference
./infer_demo --model animegan_v1_hayao_60 --image style_transfer_testimg.jpg  --device cpu
# GPU inference
./infer_demo --model animegan_v1_hayao_60 --image style_transfer_testimg.jpg  --device gpu
```

The above command works for Linux or MacOS. For SDK in Windows, refer to
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## AnimeGAN C++ Interface 

### AnimeGAN Class

```c++
fastdeploy::vision::generation::AnimeGAN(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

AnimeGAN model loading and initialization, among which model_file is the exported Paddle model file and params_file is the parameter file.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path 
> * **runtime_option**(RuntimeOption): Backend Inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict Function

> ```c++
> bool AnimeGAN::Predict(cv::Mat& image, cv::Mat* result)
> ```
>
> Model prediction interface. Input an image and output the style transfer result
>
> **Parameter**
>
> > * **image**: Input data in HWC or BGR format
> > * **result**: Image after style style transfer in BGR format

#### BatchPredict Function

> ```c++
> bool AnimeGAN::BatchPredict(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* results);
> ```
>
> Model prediction interface. Input a set of images and output style transfer results.
>
> **Parameter**
>
> > * **images**: Input data in HWC or BGR format
> > * **results**: A set of images after style transfer in BGR format.

- [Model Description](../../)
- [Python Deployment](../python)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
