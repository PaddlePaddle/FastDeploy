English | [简体中文](README_CN.md)
# PaddleDetection C++ Deployment Example

This directory provides examples that `infer_xxx.cc` fast finishes the deployment of PaddleDetection models, including PPYOLOE/PicoDet/YOLOX/YOLOv3/PPYOLO/FasterRCNN/YOLOv5/YOLOv6/YOLOv7/RTMDet on CPU/GPU and GPU accelerated by TensorRT.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
ppyoloe is taken as an example for inference deployment

mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the PPYOLOE model file and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz


# CPU inference
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 0
# GPU inference
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 1
# TensorRT Inference on GPU
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 2
# Kunlunxin XPU Inference
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 3
# Huawei Ascend Inference
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 4
```

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## PaddleDetection C++ Interface

### Model Class

PaddleDetection currently supports 6 kinds of models, including `PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`. The constructors and predictors for all 6 kinds are consistent in terms of parameters. This document takes PPYOLOE as an example to introduce its API
```c++
fastdeploy::vision::detection::PPYOLOE(
        const string& model_file,
        const string& params_file,
        const string& config_file
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

Loading and initializing PaddleDetection PPYOLOE model, where the format of model_file is as the exported ONNX model.

**Parameter**

> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path
> * **config_file**(str): •	Configuration file path, which is the deployment yaml file exported by PaddleDetection
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict Function

> ```c++
> PPYOLOE::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> Model prediction interface. Input images and output results directly.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection result, including detection box and confidence of each box. Refer to [Vision Model Prediction Result](../../../../../docs/api/vision_results/) for DetectionResult

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
