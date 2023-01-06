English | [简体中文](README_CN.md)
# PPOCRv2 C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of PPOCRv2 on CPU/GPU and GPU accelerated by TensorRT. 

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```
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
# TensorRT inference on GPU
./infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2
# Paddle-TRT inference on GPU
./infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 3
# KunlunXin XPU inference
./infer_demo ./ch_PP-OCRv2_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer ./ppocr_keys_v1.txt ./12.jpg 4
```

The above command works for Linux or MacOS. For SDK in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">


## PPOCRv2 C++ Interface 

### PPOCRv2 Class

```
fastdeploy::pipeline::PPOCRv2(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Classifier* cls_model,
                fastdeploy::vision::ocr::Recognizer* rec_model);
```

The initialization of PPOCRv2, consisting of detection, classification and recognition models

**Parameter**

> * **DBDetector**(model): Detection model in OCR
> * **Classifier**(model): Classification model in OCR
> * **Recognizer**(model): Recognition model in OCR

```
fastdeploy::pipeline::PPOCRv2(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Recognizer* rec_model);
```
The initialization of PPOCRv2, consisting of detection and recognition models (No classifier)

**Parameter**

> * **DBDetector**(model): Detection model in OCROCR中的检测模型
> * **Recognizer**(model): Recognition model in OCR

#### Predict Function

> ```  
> bool Predict(cv::Mat* img, fastdeploy::vision::OCRResult* result);
> bool Predict(const cv::Mat& img, fastdeploy::vision::OCRResult* result);
> ```
>
> Model prediction interface. Input images and output OCR prediction results
>
> **Parameter**
>
> > * **img**: Input images in HWC or BGR format
> > * **result**: OCR prediction results, including the position of the detection box from the detection model, the classification of the direction from the classification model, and the recognition result from the recognition model. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for OCRResult


## DBDetector C++ Interface 

### DBDetector Class

```
fastdeploy::vision::ocr::DBDetector(const std::string& model_file, const std::string& params_file = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const ModelFormat& model_format = ModelFormat::PADDLE);
```

DBDetector model loading and initialization. The model is in paddle format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Merely passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### The same applies to Classifier Class

### Recognizer Class
```
  Recognizer(const std::string& model_file,
             const std::string& params_file = "",
             const std::string& label_path = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const ModelFormat& model_format = ModelFormat::PADDLE);
```
For the initialization of the Recognizer class, users should input the label file required by the recognition model in the label_path parameter. Other parameters are the same as the DBDetector class

**Parameter**
> * **label_path**(str): The label path of the recognition model


### Class Member Variable
#### DBDetector Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **max_side_len**(int): The long side’s maximum size of the oriented view before detection. The long side will be resized to this size when exceeding the value. And the short side will be scaled in equal proportion. Default 960
> > * **det_db_thresh**(double): The binarization threshold of the prediction image from DB models. Default 0.3
> > * **det_db_box_thresh**(double): The threshold for the output box of DB models, below which the predicted box is discarded. Default 0.6 
> > * **det_db_unclip_ratio**(double): The expansion ratio of the DB model output box. Default 1.5
> > * **det_db_score_mode**(string): The way to calculate the average score of the text box in DB post-processing. Default slow, which is identical to the calculation of the polygon area’s average score
> > * **use_dilation**(bool): Whether to expand the feature map from the detection. Default False

#### Classifier Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **cls_thresh**(double): The input image will be flipped when the score output by the classification model exceeds this threshold. Default 0.9

## Other Documents

- [PPOCR Model Description](../../)
- [PPOCRv2 Python Deployment](../python)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
