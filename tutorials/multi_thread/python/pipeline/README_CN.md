[English](README.md) | 简体中文
# PPOCR模型 Python多线程/进程部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`multi_thread_process_ocr.py`快速完成PPOCRv3在CPU/GPU，以及GPU上通过TensorRT加速部署的多线程/进程示例。执行如下脚本即可完成


```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/tutorials/multi_thread/python/pipeline

# 下载模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU多线程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device cpu --thread_num 1
# CPU多进程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device cpu --use_multi_process True --process_num 1

# GPU多线程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --thread_num 1
# GPU多进程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --use_multi_process True --process_num 1

# GPU上使用TensorRT多线程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --backend trt --thread_num 1
# GPU上使用TensorRT多进程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device gpu --backend trt --use_multi_process True --process_num 1

# 昆仑芯XPU多线程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device kunlunxin --thread_num 1
# 昆仑芯XPU多进程推理
python multi_thread_process_ocr.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image_path 12.jpg --device kunlunxin --use_multi_process True --process_num 1

```
>> **注意**: `--image_path` 可以输入图片文件夹的路径

运行完成后返回结果如下所示
```
thread: 0 , result:  det boxes: [[42,413],[483,391],[484,428],[43,450]]rec text: 上海斯格威铂尔大酒店 rec score:0.949773 cls label: 0 cls score: 1.000000
det boxes: [[187,456],[399,448],[400,480],[188,488]]rec text: 打浦路15号 rec score:0.910265 cls label: 0 cls score: 1.000000
det boxes: [[23,507],[513,488],[515,529],[24,548]]rec text: 绿洲仕格维花园公寓 rec score:0.934239 cls label: 0 cls score: 1.000000
det boxes: [[74,553],[427,542],[428,571],[75,582]]rec text: 打浦路252935号 rec score:0.872207 cls label: 0 cls score: 1.000000
```
