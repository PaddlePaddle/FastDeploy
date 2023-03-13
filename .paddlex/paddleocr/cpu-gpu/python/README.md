# PaddleOCR CPU-GPU Python部署示例
本目录下提供`infer.py`快速完成PP-OCRv3在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例.

PaddleOCR支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署OCR模型

## 1. 运行部署示例
```bash
# 找到部署包内的模型，这里测试的模型包括检测模型、分类模型和识别模型，如果只导出了其中的某个模型，则其他模型可用预训练模型进行测试
# 例如只导出了检测模型，则分类和识别模型可用预训练模型ch_ppocr_mobile_v2.0_cls_infer.tar和识别模型ch_PP-OCRv3_rec_infer.tar

# 可用于测试的预训练模型，可替换为自己训练的模型
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar

# 下载预测图片与字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 运行部署示例
# 在CPU上使用Paddle Inference推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu --backend paddle
# 在CPU上使用OenVINO推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu --backend openvino
# 在CPU上使用ONNX Runtime推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu --backend ort
# 在CPU上使用Paddle Lite推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu --backend pplite
# 在GPU上使用Paddle Inference推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --backend paddle
# 在GPU上使用Paddle TensorRT推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --backend pptrt
# 在GPU上使用ONNX Runtime推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --backend ort
# 在GPU上使用Nvidia TensorRT推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --backend trt

# 同时, FastDeploy提供文字检测,文字分类,文字识别三个模型的单独推理,
# 有需要的用户, 请准备合适的图片, 同时根据自己的需求, 参考infer.py来配置自定义硬件与推理后端.

# 在CPU上,单独使用文字检测模型部署
python infer_det.py --det_model ch_PP-OCRv3_det_infer  --image 12.jpg --device cpu

# 在CPU上,单独使用文字方向分类模型部署
python infer_cls.py --cls_model ch_ppocr_mobile_v2.0_cls_infer --image 12.jpg --device cpu

# 在CPU上,单独使用文字识别模型部署
python infer_rec.py  --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu

```

## 2. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--det_model|指定检测模型文件夹所在的路径|None|
|--cls_model|指定分类模型文件夹所在的路径|None|
|--rec_model|指定识别模型文件夹所在的路径|None|
|--rec_label_file|识别模型所需label所在的路径|None|
|--image|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--device_id|使用gpu时, 指定设备号|0|
|--backend|部署模型时使用的后端, 支持的值为`[paddle,pptrt,pplite,ort,openvino,trt]` |paddle|

## 3. 更多指南

### 3.1 如何使用Python部署PP-OCRv2系列模型.
本目录下的`infer.py`代码是以PP-OCRv3模型为例, 如果用户有使用PP-OCRv2的需求, 只需要按照下面所示的方式, 来创建PP-OCRv2并使用.

```python
# 此行为创建PP-OCRv3模型的代码
ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
# 只需要将PPOCRv3改为PPOCRv2,即可创造PPOCRv2模型, 同时, 后续的接口均使用ppocr_v2来调用
ppocr_v2 = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 如果用户在部署PP-OCRv2时, 需要使用TensorRT推理, 还需要改动Rec模型的TensorRT的输入shape.
# 建议如下修改, 需要把 H 维度改为32, W 纬度按需修改.
rec_option.set_trt_input_shape("x", [1, 3, 32, 10],
                                       [args.rec_bs, 3, 32, 320],
                                       [args.rec_bs, 3, 32, 2304])
```

### 3.2 如何在PP-OCRv2/v3系列模型中, 关闭文字方向分类器的使用.

在PP-OCRv3/v2中, 文字方向分类器是可选的, 用户可以按照以下方式, 来决定自己是否使用方向分类器.
```python
# 使用 Cls 模型
ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 不使用 Cls 模型
ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=None, rec_model=rec_model)
```
### 3.3 如何修改前后处理超参数.
在示例代码中, 我们展示出了修改前后处理超参数的接口,并设置为默认值,其中, FastDeploy提供的超参数的含义与文档[PaddleOCR推理模型参数解释](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/inference_args.md)是相同的. 如果用户想要进行更多定制化的开发, 请阅读[PP-OCR系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/ocr.html)

```python
# 设置检测模型的max_side_len
det_model.preprocessor.max_side_len = 960
# 其他...
```
