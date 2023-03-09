English | [简体中文](README_CN.md)
# PaddleClas C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of PaddleClas models on CPU/GPU and GPU accelerated by TensorRT.

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

Taking ResNet50_vd inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0)  is required to support this model.

```bash
# Find the model directory in the package, e.g. ResNet50

# Prepare a test image, e.g. test.jpg

# CPU inference
./infer_demo ResNet50 test.jpg 0
# GPU inference
./infer_demo ResNet50 test.jpg 1
# TensorRT inference on GPU
./infer_demo ResNet50 test.jpg 2
# IPU inference
./infer_demo ResNet50 test.jpg 3
# KunlunXin XPU inference
./infer_demo ResNet50 test.jpg 4
# Ascend inference
./infer_demo ResNet50 test.jpg 5
```

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
> > * **result**: The classification result, including label_id, and the corresponding confidence. Refer to [Visual Model Prediction Results](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/classification_result.md) for the description of ClassifyResult
> > * **topk**(int): Return the topk classification results with the highest prediction probability. Default 1
