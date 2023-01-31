English | [简体中文](README_CN.md)
# YOLOv5Cls C++ Deployment Example

This directory provides examples that ` infer.cc` fast finishes the deployment of YOLOv5Cls models on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy  Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

Taking CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0)  is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the official converted yolov5 model file and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-cls.onnx
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU inference
./infer_demo yolov5n-cls.onnx 000000014439.jpg 0
# GPU inference
./infer_demo yolov5n-cls.onnx 000000014439.jpg 1
# TensorRT Inference on GPU
./infer_demo yolov5n-cls.onnx 000000014439.jpg 2
```

The result returned after running is as follows
```bash
ClassifyResult(
label_ids: 265,
scores: 0.196327,
)
```

The above command works for Linux or MacOS. Refer to:  
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md) for SDK use-pattern in Windows.

## YOLOv5Cls C++ Interface 

### YOLOv5Cls Class

```c++
fastdeploy::vision::classification::YOLOv5Cls(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv5Cls model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Only passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict Function

> ```c++
> YOLOv5Cls::Predict(cv::Mat* im, int topk = 1)
> ```
>
> Model prediction interface. Input images and output classification topk results directly.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format
> > * **topk**(int): Return the topk classification results with the highest prediction probability. Default 1

> **Return**
>
> > Return `fastdeploy.vision.ClassifyResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.


## Other Documents

- [YOLOv5Cls Model Description](..)
- [YOLOv5Cls Python Deployment](../python)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
