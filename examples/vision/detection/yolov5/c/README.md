English | [简体中文](README_CN.md)
# YOLOv5 C Deployment Example

This directory provides `infer.c` to finish the deployment of YOLOv5 on CPU/GPU.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4) is required to support this model.

```bash
# 1. # Download the YOLOv5 model file and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
./infer_demo yolov5s.onnx 000000014439.jpg 0
# GPU inference
./infer_demo yolov5s.onnx 000000014439.jpg 1
```

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## YOLOv5 C Interface

### RuntimeOption

```c
FD_C_RuntimeOptionWrapper* FD_C_CreateRuntimeOptionWrapper()
```

> Create a RuntimeOption object, and return a pointer to manipulate it.
>
> **Return**
>
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

FD_C_YOLOv5Wrapper* FD_C_CreateYOLOv5Wrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* runtime_option,
    const FD_C_ModelFormat model_format)

```

> Create a YOLOv5 model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path，when model format is onnx，this can be empty string
> * **runtime_option**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format.
>
> **Return**
> * **fd_c_yolov5_wrapper**(FD_C_YOLOv5Wrapper*): Pointer to manipulate YOLOv5 object.


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
FD_C_Bool FD_C_YOLOv5WrapperPredict(
    __fd_take FD_C_YOLOv5Wrapper* fd_c_yolov5_wrapper, FD_C_Mat img,
    FD_C_DetectionResult* fd_c_detection_result)
```
>
> Predict an image, and generate detection result.
>
> **Params**
> * **fd_c_yolov5_wrapper**(FD_C_YOLOv5Wrapper*): Pointer to manipulate YOLOv5 object.
> * **img**（FD_C_Mat）: pointer to cv::Mat object, which can be obained by FD_C_Imread interface
> * **fd_c_detection_result**FD_C_DetectionResult*): Detection result, including detection box and confidence of each box. Refer to [Vision Model Prediction Result](../../../../../docs/api/vision_results/) for DetectionResults


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
