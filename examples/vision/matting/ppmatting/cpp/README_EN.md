English | [简体中文](README.md)
# PP-Matting C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of PP-Matting on CPU/GPU and GPU accelerated by TensorRT. 
Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy  Precompiled Library](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

Taking the PP-Matting inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy  Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download PP-Matting model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz
tar -xvf PP-Matting-512.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg


# CPU inference
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 0
# GPU inference
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 1
# TensorRT inference on GPU
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 2
# kunlunxin XPU inference
./infer_demo PP-Matting-512 matting_input.jpg matting_bgr.jpg 3
```

The visualized result after running is as follows
<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## PP-Matting C++ Interface 

### PPMatting Class

```c++
fastdeploy::vision::matting::PPMatting(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PP-Matting model loading and initialization, among which model_file is the exported Paddle model format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

#### Predict Function

> ```c++
> PPMatting::Predict(cv::Mat* im, MattingResult* result)
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


- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
