English | [简体中文](README_CN.md)
# PaddleClas C Deployment Example

This directory provides examples that `infer.c` fast finishes the deployment of PaddleClas models on CPU/GPU.

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

Taking ResNet50_vd inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4)  is required to support this model.

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
```

The above command works for Linux or MacOS. Refer to
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md) for SDK use-pattern in Windows

## PaddleClas C Interface

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
> 开启GPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Pointer to manipulate RuntimeOption object.
> * **gpu_id**(int): gpu id


### Model

```c

FD_C_PaddleClasModelWrapper* FD_C_CreatePaddleClasModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* runtime_option,
    const FD_C_ModelFormat model_format)

```

> Create a PaddleClas model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(const char*): Model file path
> * **params_file**(const char*): Parameter file path
> * **config_file**(const char*): Configuration file path, which is the deployment yaml file exported by PaddleClas.
> * **runtime_option**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format. Paddle format by default
>
> **Return**
> * **fd_c_ppclas_wrapper**(FD_C_PaddleClasModelWrapper*): Pointer to manipulate PaddleClas object.


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
FD_C_Bool FD_C_PaddleClasModelWrapperPredict(
    __fd_take FD_C_PaddleClasModelWrapper* fd_c_ppclas_wrapper, FD_C_Mat img,
    FD_C_ClassifyResultWrapper* fd_c_ppclas_result_wrapper)
```
>
> Predict an image, and generate classification result.
>
> **Params**
> * **fd_c_ppclas_wrapper**(FD_C_PaddleClasModelWrapper*): pointer to manipulate PaddleClas object
> * **img**（FD_C_Mat）: pointer to cv::Mat object, which can be obained by FD_C_Imread interface
> * **result**（FD_C_ClassifyResultWrapper*): The classification result, including label_id, and the corresponding confidence. Refer to [Visual Model Prediction Results](../../../../../docs/api/vision_results/) for the description of ClassifyResult


#### Result

```c
FD_C_ClassifyResultWrapper* FD_C_CreateClassifyResultWrapper();
```
>
> Create a ClassifyResult object to keep the classification result，return a pointer to manipulate it.
>
> **Return**
> * **fd_c_classify_result_wrapper**(FD_C_ClassifyResultWrapper*): pointer to manipulate ClassifyResult object



```c
FD_C_ClassifyResult* FD_C_ClassifyResultWrapperGetData(
     FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper)
```
>
>
>
> **Params**
> * **fd_c_classify_result_wrapper**(FD_C_ClassifyResultWrapper*): pointer to manipulate ClassifyResult object
>
> **Return**
> * **fd_c_classify_result**(FD_C_ClassifyResult*): pointer to C ClassifyResult structure


- [Model Description](../../)
- [Python Deployment](../python)
- [Visual Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
