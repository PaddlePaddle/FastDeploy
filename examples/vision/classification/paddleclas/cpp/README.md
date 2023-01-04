English | [简体中文](README_CN.md)
# PaddleClas C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of PaddleClas models on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

Taking ResNet50_vd inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0)  is required to support this model.

```bash
mkdir build
cd build
# Download FastDeploy precompiled library. Users can choose your appropriate version in the`FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download ResNet50_vd model file and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU inference
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0
# GPU inference
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1
# TensorRT inference on GPU
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 2
# IPU inference
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 3
# KunlunXin XPU inference
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 4
```

The above command works for Linux or MacOS. Refer to 
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md) for SDK use-pattern in Windows

## PaddleClas C++ Interface 

### PaddleClas Class

```c++
fastdeploy::vision::classification::PaddleClasModel(
        const string& model_file,
        const string& params_file,
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PaddleClas model loading and initialization, where model_file and params_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path 
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict function

> ```c++
> PaddleClasModel::Predict(cv::Mat* im, ClassifyResult* result, int topk = 1)
> ```
>
> Model prediction interface. Input images and output results directly.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: The classification result, including label_id, and the corresponding confidence. Refer to [Visual Model Prediction Results](../../../../../docs/api/vision_results/) for the description of ClassifyResult
> > * **topk**(int): Return the topk classification results with the highest prediction probability. Default 1


- [Model Description](../../)
- [Python Deployment](../python)
- [Visual Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
