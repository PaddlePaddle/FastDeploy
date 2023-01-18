English | [简体中文](README_CN.md)
# PPOCRv3 Python multi-thread/multi-process Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides example file `multi_thread_process_ocr.py` to fast deploy multi-thread/multi-process ResNet50_vd on CPU/GPU and GPU accelerated by TensorRT. The script is as follows


```bash
# Download deployment example code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/tutorials/multi_thread/python/pipeline

# Download model, image, and dictionary files
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU multi-thread inference
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device cpu --thread_num 1
# CPU multi-process inference
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device cpu --use_multi_process True --process_num 1

# GPU multi-thread inference
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --thread_num 1
# GPU multi-process inference
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --use_multi_process True --process_num 1

# Use TensorRT multi-thread inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --backend trt --thread_num 1
# Use TensorRT multi-process inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --backend trt --use_multi_process True --process_num 1

# KunlunXin XPU multi-thread inference
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device kunlunxin --thread_num 1
# KunlunXin XPU multi-process inference
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device kunlunxin --use_multi_process True --process_num 1

```
>> **Notice**: `--image_path` can be the path of the pictures folder

The result returned after running is as follows
```
thread: 0 , result:  det boxes: [[42,413],[483,391],[484,428],[43,450]]rec text: 上海斯格威铂尔大酒店 rec score:0.949773 cls label: 0 cls score: 1.000000
det boxes: [[187,456],[399,448],[400,480],[188,488]]rec text: 打浦路15号 rec score:0.910265 cls label: 0 cls score: 1.000000
det boxes: [[23,507],[513,488],[515,529],[24,548]]rec text: 绿洲仕格维花园公寓 rec score:0.934239 cls label: 0 cls score: 1.000000
det boxes: [[74,553],[427,542],[428,571],[75,582]]rec text: 打浦路252935号 rec score:0.872207 cls label: 0 cls score: 1.000000
```
