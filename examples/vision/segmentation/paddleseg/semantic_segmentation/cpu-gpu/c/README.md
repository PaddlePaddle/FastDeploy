English | [简体中文](README_CN.md)
# PaddleSeg C Deployment Example

This directory provides `infer.c` to finish the deployment of PaddleSeg on CPU/GPU.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/build_and_install/download_prebuilt_libraries.md)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4) is required to support this model.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download model, image files
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png


# CPU inference
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 0
# GPU inference
./infer_demo PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer cityscapes_demo.png 1
```

The above command works for Linux or MacOS. For SDK in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/use_sdk_on_windows.md)

The visualized result after running is as follows

<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>


## PaddleSeg C Interface

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
FD_C_PaddleSegWrapper* FD_C_CreatePaddleSegWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```

> Create a PaddleSeg model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(const char*): Model file path
> * **params_file**(const char*): Parameter file path
> * **config_file**(const char*): config file path
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format
>
> **Return**
>
> * **fd_c_ppseg_wrapper**(FD_C_PaddleSegWrapper*): Pointer to manipulate PaddleSeg object.



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
FD_C_Bool FD_C_PaddleSegWrapperPredict(
    FD_C_PaddleSegWrapper* fd_c_ppseg_wrapper,
    FD_C_Mat img,
    FD_C_SegmentationResult* result)
```
>
> Predict an image, and generate result.
>
> **Params**
> * **fd_c_ppseg_wrapper**(FD_C_PaddleSegWrapper*): Pointer to manipulate PaddleSeg object.
> * **img**（FD_C_Mat）: pointer to cv::Mat object, which can be obained by FD_C_Imread interface
> * **result**(FD_C_SegmentationResult*): Segmentation prediction results, Refer to [Vision Model Prediction Results](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/) for SegmentationResult


#### Result

```c
FD_C_Mat FD_C_VisSegmentation(FD_C_Mat im,
                              FD_C_SegmentationResult* result,
                              float weight)
```
>
> Visualize segmentation results and return visualization image.
>
> **Params**
> * **im**(FD_C_Mat): pointer to input image
> * **segmentation_result**(FD_C_SegmentationResult*): pointer to C FD_C_SegmentationResult structure
> * **weight**(float): weight transparent weight of visualized result image
>
> **Return**
> * **vis_im**(FD_C_Mat): pointer to visualization image.


## Other Documents

- [PPSegmentation Model Description](../../)
- [PaddleSeg Python Deployment](../python)
- [Model Prediction Results](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/)
- [How to switch the model inference backend engine](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/how_to_change_backend.md)
