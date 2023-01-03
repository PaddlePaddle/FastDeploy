English | [简体中文](README.md)
# PPOCRv2 Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of PPOCRv2 on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```

# Download model, image, and dictionary files
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
tar -xvf ch_PP-OCRv2_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
tar -xvf ch_PP-OCRv2_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt


# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vison/ocr/PP-OCRv2/python/

# CPU inference
python infer.py --det_model ch_PP-OCRv2_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv2_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu
# GPU inference
python infer.py --det_model ch_PP-OCRv2_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv2_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --det_model ch_PP-OCRv2_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv2_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --backend trt
# KunlunXin XPU inference
python infer.py --det_model ch_PP-OCRv2_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv2_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device kunlunxin
```

The visualized result after running is as follows
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">

## PPOCRv2 Python Interface  

```
fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
```
To initialize PPOCRv2, the input parameters correspond to detection model, classification model, and recognition model. Among them, cls_model is optional. It can be set to None if there is no demand

**Parameter**

> * **det_model**(model): Detection model in OCR
> * **cls_model**(model): Classification model in OCR
> * **rec_model**(model): Recognition model in OCR

### predict function

> ```
> result = ppocr_v2.predict(im)
> ```
>
> Model prediction interface. Input one image.
>
> **Parameter**
>
> > * **im**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return the `fastdeploy.vision.OCRResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description.



## DBDetector Python Interface 

### DBDetector Class

```
fastdeploy.vision.ocr.DBDetector(model_file, params_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

DBDetector model loading and initialization. The model is in paddle format.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Merely passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. PADDLE format by default

### The same applies to the Classifier class

### Recognizer Class
```
fastdeploy.vision.ocr.Recognizer(rec_model_file,rec_params_file,rec_label_file,
                                  runtime_option=rec_runtime_option,model_format=ModelFormat.PADDLE)
```
To initialize the Recognizer class, users need to input the label file path required by the recognition model in the rec_label_file parameter. Other parameters are the same as those of DBDetector class

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
- [PPOCRv2 C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
