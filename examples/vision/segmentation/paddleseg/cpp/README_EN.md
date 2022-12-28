English | [简体中文](README.md)
# PaddleSeg C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of Unet on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

【Attention】For the deployment of **PP-Matting**、**PP-HumanMatting** and **ModNet**, refer to [Matting Model Deployment](../../../matting)

Taking the inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.0 or above (x.x.x>=1.0.0) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download Unet model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz
tar -xvf Unet_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png


# CPU inference
./infer_demo Unet_cityscapes_without_argmax_infer cityscapes_demo.png 0
# GPU inference
./infer_demo Unet_cityscapes_without_argmax_infer cityscapes_demo.png 1
# TensorRT inference on GPU
./infer_demo Unet_cityscapes_without_argmax_infer cityscapes_demo.png 2
# kunlunxin XPU inference
./infer_demo Unet_cityscapes_without_argmax_infer cityscapes_demo.png 3
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## PaddleSeg C++ Interface 

### PaddleSeg Class

```c++
fastdeploy::vision::segmentation::PaddleSegModel(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PaddleSegModel model loading and initialization, among which model_file is the exported Paddle model format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict Function

> ```c++
> PaddleSegModel::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: The segmentation result, including the predicted label of the segmentation and the corresponding probability of the label. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of SegmentationResult

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **is_vertical_screen**(bool): For PP-HumanSeg models, the input image is portrait, height greater than a width, by setting this parameter to`true`

#### Post-processing Parameter
> > * **apply_softmax**(bool): The `apply_softmax` parameter is not specified when the model is exported. Set this parameter to `true` to normalize the probability result (score_map) of the predicted output segmentation label (label_map)

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
