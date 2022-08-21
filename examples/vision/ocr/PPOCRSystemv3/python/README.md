# PPOCRSystemv3 Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/the%20software%20and%20hardware%20requirements.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`infer.py`快速完成PPOCRSystemv3在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```

# 下载模型,图片和label文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar

wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/doc/imgs/12.jpg

wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/ppocr/utils/ppocr_keys_v1.txt


#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vison/ocr/PPOCRSystemv3/python/

# CPU推理
python infer_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu
# GPU推理
python infer_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu
# GPU上使用TensorRT推理
python infer_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --det_use_trt True --cls_use_trt True --rec_use_trt True

```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184309358-d803347a-8981-44b6-b589-4608021ad0f4.jpg">

## PPOCRSystemv3 Python接口

```
fastdeploy.vision.ocr.PPOCRSystemv3(ocr_det = det_model._model, ocr_cls = cls_model._model, ocr_rec = rec_model._model)
```

PPOCRSystemv3的初始化,输入的参数是检测模型，分类模型和识别模型

**参数**

> * **ocr_det**(model): OCR中的检测模型
> * **ocr_cls**(model): OCR中的分类模型
> * **ocr_rec**(model): OCR中的识别模型

### predict函数

> ```
> result = PPOCRSystemv3.predict(img_list)
> ```
>
> 模型预测接口，输入的是一个可包含多个图像的list
>
> **参数**
>
> > * **img_list**(list[np.ndarray]): 输入数据的list，每张图片注意需为HWC，BGR格式
> > * **result**(float): OCR结果,包括由检测模型输出的检测框位置,分类模型输出的方向分类,以及识别模型输出的识别结果,

> **返回**
>
> > 返回`fastdeploy.vision.OCRResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)



## DBDetector Python接口

### DBDetector类

```
fastdeploy.vision.ocr.DBDetector(model_file, params_file, runtime_option=None, model_format=Frontend.PADDLE)
```

DBDetector模型加载和初始化，其中模型为paddle模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为PADDLE格式

### Classifier类与DBDetector类相同

### Recognizer类
```
fastdeploy.vision.ocr.Recognizer(rec_label_file,rec_model_file,rec_params_file,
                                  runtime_option=rec_runtime_option,model_format=Frontend.PADDLE)
```
Recognizer类初始化时,需要在rec_label_file参数中,输入识别模型所需的label文件路径，其他参数均与DBDetector类相同

**参数**
> * **label_path**(str): 识别模型的label文件路径



### 类成员变量
#### DBDetector预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[640, 640]
> > * **padding_value**(vector&lt;float&gt;): 通过此参数可以修改图片在resize时候做填充(padding)的值, 包含三个浮点型元素, 分别表示三个通道的值, 默认值为[114, 114, 114]
> > * **is_no_pad**(bool): 通过此参数让图片是否通过填充的方式进行resize, `is_no_pad=ture` 表示不使用填充的方式，默认值为`is_no_pad=false`
> > * **is_mini_pad**(bool): 通过此参数可以将resize之后图像的宽高这是为最接近`size`成员变量的值, 并且满足填充的像素大小是可以被`stride`成员变量整除的。默认值为`is_mini_pad=false`
> > * **stride**(int): 配合`stris_mini_pad`成员变量使用, 默认值为`stride=32`

#### DBDetector预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **max_side_len**(int): 检测算法前向时图片长边的最大尺寸，当长边超出这个值时会将长边resize到这个大小，短边等比例缩放,默认为960
> > * **det_db_thresh**(double): DB模型输出预测图的二值化阈值，默认为0.3
> > * **det_db_box_thresh**(double): DB模型输出框的阈值，低于此值的预测框会被丢弃，默认为0.6
> > * **det_db_unclip_ratio**(double): DB模型输出框扩大的比例，默认为1.5
> > * **det_db_score_mode**(string):DB后处理中计算文本框平均得分的方式,默认为slow，即求polygon区域的平均分数的方式
> > * **use_dilation**(bool):是否对检测输出的feature map做膨胀处理,默认为Fasle

#### Classifier预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **cls_thresh**(double): 当分类模型输出的得分超过此阈值，输入的图片将被翻转，默认为0.9



## 其它文档

- [YOLOv5 模型介绍](..)
- [YOLOv5 C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
