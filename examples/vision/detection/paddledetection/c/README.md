English | [简体中文](README_CN.md)
# PaddleDetection C Deployment Example

This directory provides examples that `infer_xxx.c` fast finishes the deployment of PaddleDetection models, including PPYOLOE on CPU/GPU.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4) is required to support this model.

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
```

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## PaddleDetection C Interface

### RuntimeOption

```c
FD_C_RuntimeOptionWrapper* FD_C_CreateRuntimeOptionWrapper()
```

> Create a RuntimeOption object, and return a pointer to manipulate it.
>
> **Return**
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Pointer to manipulate RuntimeOption object.


```c
void FD_C_RuntimeOptionWrapperUseCpu(
     FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper)
```

> Enable Cpu inference.
>
> **Params**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Pointer to manipulate RuntimeOption object.

```c
void FD_C_RuntimeOptionWrapperUseGpu(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int gpu_id)
```
> Enable Gpu inference.
>
> **Params**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Pointer to manipulate RuntimeOption object.
> * **gpu_id**(int): gpu id


### Model

```c

FD_C_PPYOLOEWrapper* FD_C_CreatePPYOLOEWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* runtime_option,
    const FD_C_ModelFormat model_format)

```

> Create a PPYOLOE model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(const char*): Model file path
> * **params_file**(const char*): Parameter file path
> * **config_file**(const char*): Configuration file path, which is the deployment yaml file exported by PaddleDetection
> * **runtime_option**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format. FD_C_ModelFormat_PADDLE format by default
>
> **Return**
> * **fd_c_ppyoloe_wrapper**(FD_C_PPYOLOEWrapper*): Pointer to manipulate PPYOLOE object.


#### Read and write image

```c
FD_C_Mat FD_C_Imread(const char* imgpath)
```

> Read an image, and return a pointer to cv::Mat.
>
> **Params**
>
> * **imgpath**(const char*): image path
>
> **Return**
>
> * **imgmat**(FD_C_Mat): pointer to cv::Mat object which holds the image.


```c
FD_C_Bool FD_C_Imwrite(const char* savepath,  FD_C_Mat img);
```

> Write image to a file.
>
> **Params**
>
> * **savepath**(const char*): save path
> * **img**(FD_C_Mat): pointer to cv::Mat object
>
> **Return**
>
> * **result**(FD_C_Bool): bool to indicate success or failure


#### Prediction

```c
FD_C_Bool FD_C_PPYOLOEWrapperPredict(
    __fd_take FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper, FD_C_Mat img,
    FD_C_DetectionResult* fd_c_detection_result)
```
>
> Predict an image, and generate detection result.
>
> **Params**
> * **fd_c_ppyoloe_wrapper**(FD_C_PPYOLOEWrapper*): pointer to manipulate PPYOLOE object
> * **img**（FD_C_Mat）: pointer to cv::Mat object, which can be obained by FD_C_Imread interface
> * **fd_c_detection_result**FD_C_DetectionResult*): Detection result, including detection box and confidence of each box. Refer to [Vision Model Prediction Result](../../../../../docs/api/vision_results/) for DetectionResult


#### Result

```c
FD_C_Mat FD_C_VisDetection(FD_C_Mat im, FD_C_DetectionResult* fd_detection_result,
                  float score_threshold, int line_size, float font_size);
```
>
> Visualize detection results and return visualization image.
>
> **Params**
> * **im**(FD_C_Mat): pointer to input image
> * **fd_detection_result**(FD_C_DetectionResult*): pointer to C DetectionResult structure
> * **score_threshold**(float): score threshold
> * **line_size**(int): line size
> * **font_size**(float): font size
>
> **Return**
> * **vis_im**(FD_C_Mat): pointer to visualization image.


- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
