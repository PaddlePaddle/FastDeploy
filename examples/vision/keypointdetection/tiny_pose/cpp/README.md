English | [简体中文](README_CN.md)
# PP-TinyPose C++ Deployment Example

This directory provides the `Multi-person keypoint detection in a single image` example that `pptinypose_infer.cc` fast finishes the deployment of PP-TinyPose on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
>> **Attention**: PP-Tinypose single model currently supports single-person keypoint detection in a single image. Therefore, the input image should contain one person only or should be cropped. For multi-person keypoint detection, refer to [PP-TinyPose Pipeline](../../det_keypoint_unite/cpp/README.md)

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)


Taking the inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0)  is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download PP-TinyPose model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/hrnet_demo.jpg


# CPU inference
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 0
# GPU inference
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 1
# TensorRT inference on GPU
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 2
# KunlunXin XPU inference
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 3
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196386764-dd51ad56-c410-4c54-9580-643f282f5a83.jpeg", width=359px, height=423px />
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## PP-TinyPose C++ Interface 

### PP-TinyPose Class

```c++
fastdeploy::vision::keypointdetection::PPTinyPose(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PPTinyPose  model loading and initialization, among which model_file is the exported Paddle model format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict function

> ```c++
> PPTinyPose::Predict(cv::Mat* im, KeyPointDetectionResult* result)
> ```
>
> Model prediction interface. Input images and output keypoint detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Keypoint detection results, including coordinates and the corresponding probability value. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of KeyPointDetectionResult

### Class Member Property
#### Post-processing Parameter
> > * **use_dark**(bool): Whether to use DARK for post-processing. Refer to [Reference Paper](https://arxiv.org/abs/1910.06278)

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
