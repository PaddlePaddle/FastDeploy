English | [简体中文](README_CN.md)
# PPOCRv2 C Deployment Example

This directory provides `infer.c` to finish the deployment of PPOCRv2 on CPU/GPU.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 1.0.4 or above (x.x.x>=1.0.4) is required to support this model.


```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j


# Download model, image, and dictionary files
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
tar -xvf ch_PP-OCRv2_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
tar -xvf ch_PP-OCRv2_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU inference
./infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU inference
./infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
```

The above command works for Linux or MacOS. For SDK in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

The visualized result after running is as follows


<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">


## PPOCRv2 C Interface

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

FD_C_DBDetectorWrapper* FD_C_CreateDBDetectorWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)

```

> Create a DBDetector model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(const char*): Model file path
> * **params_file**(const char*): Parameter file path
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format.
>
> **Return**
> * **fd_c_dbdetector_wrapper**(FD_C_DBDetectorWrapper*): Pointer to manipulate DBDetector object.

```c
FD_C_ClassifierWrapper* FD_C_CreateClassifierWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```
> Create a Classifier model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(const char*): Model file path
> * **params_file**(const char*): Parameter file path
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format.
>
> **Return**
>
> * **fd_c_classifier_wrapper**(FD_C_ClassifierWrapper*): Pointer to manipulate Classifier object.

```c
FD_C_RecognizerWrapper* FD_C_CreateRecognizerWrapper(
    const char* model_file, const char* params_file, const char* label_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```
> Create a Recognizer model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **model_file**(const char*):  Model file path
> * **params_file**(const char*): Parameter file path
> * **label_path**(const char*): Label file path
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(FD_C_ModelFormat): Model format.
>
> **Return**
> * **fd_c_recognizer_wrapper**(FD_C_RecognizerWrapper*): Pointer to manipulate Recognizer object.

```c
FD_C_PPOCRv2Wrapper* FD_C_CreatePPOCRv2Wrapper(
    FD_C_DBDetectorWrapper* det_model,
    FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model
)
```
> Create a PPOCRv2 model object, and return a pointer to manipulate it.
>
> **Params**
>
> * **det_model**(FD_C_DBDetectorWrapper*): DBDetector model
> * **cls_model**(FD_C_ClassifierWrapper*): Classifier model
> * **rec_model**(FD_C_RecognizerWrapper*): Recognizer model
>
> **Return**
>
> * **fd_c_ppocrv2_wrapper**(FD_C_PPOCRv2Wrapper*): Pointer to manipulate PPOCRv2 object.



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
FD_C_Bool FD_C_PPOCRv2WrapperPredict(
    FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper,
    FD_C_Mat img,
    FD_C_OCRResult* result)
```
>
> Predict an image, and generate result.
>
> **Params**
> * **fd_c_ppocrv2_wrapper**(FD_C_PPOCRv2Wrapper*): Pointer to manipulate PPOCRv2 object.
> * **img**（FD_C_Mat）: pointer to cv::Mat object, which can be obained by FD_C_Imread interface
> * **result**(FD_C_OCRResult*): OCR prediction results, including the position of the detection box from the detection model, the classification of the direction from the classification model, and the recognition result from the recognition model. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for OCRResult

#### Result

```c
FD_C_Mat FD_C_VisOcr(FD_C_Mat im, FD_C_OCRResult* ocr_result)
```
>
> Visualize OCR results and return visualization image.
>
> **Params**
> * **im**(FD_C_Mat): pointer to input image
> * **ocr_result**(FD_C_OCRResult*): pointer to C FD_C_OCRResult structure
>
> **Return**
> * **vis_im**(FD_C_Mat): pointer to visualization image.




## Other Documents

- [PPOCR Model Description](../../)
- [PPOCRv2 Python Deployment](../python)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
