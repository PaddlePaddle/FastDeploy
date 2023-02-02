[English](README.md) | 中文
# PPOCRv3模型 C++多线程部署示例

本目录下提供`multi_thread_ocr.cc`快速完成PPOCRv3系列模型在CPU/GPU，以及GPU上通过TensorRT加速多线程部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上ResNet50_vd推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j


# 下载模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU推理
./multi_thread_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0 1
# GPU推理
./multi_thread_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1 1
# GPU上TensorRT推理
./multi_thread_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2 1
# GPU上Paddle-TRT推理
./multi_thread_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 3 1
# 昆仑芯XPU推理
./multi_thread_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 4 1
>> **注意**: 最后一位数字表示线程数
```
以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../docs/cn/faq/use_sdk_on_windows.md)

运行完成后返回结果如下所示
```
Thread Id: 0
det boxes: [[42,413],[483,391],[484,428],[43,450]]rec text: 上海斯格威铂尔大酒店 rec score:0.980085 cls label: 0 cls score: 1.000000
det boxes: [[187,456],[399,448],[400,480],[188,488]]rec text: 打浦路15号 rec score:0.964993 cls label: 0 cls score: 1.000000
det boxes: [[23,507],[513,488],[515,529],[24,548]]rec text: 绿洲仕格维花园公寓 rec score:0.993727 cls label: 0 cls score: 1.000000
det boxes: [[74,553],[427,542],[428,571],[75,582]]rec text: 打浦路252935号 rec score:0.947723 cls label: 0 cls score: 1.000000
```
