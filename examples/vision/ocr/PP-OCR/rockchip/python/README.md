[English](README_CN.md) | 简体中文
# PP-OCRv3 RKNPU2 Python部署示例
本目录下提供`infer.py`, 供用户完成PP-OCRv3在RKNPU2的部署.


## 1. 部署环境准备
在部署前，需确认以下两个步骤
- 1. 在部署前，需自行编译基于RKNPU2的Python预测库，参考文档[RKNPU2部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)
- 2. 同时请用户参考[FastDeploy RKNPU2资源导航](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rknpu2.md)

## 2.部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载或转换所需模型.

## 3.运行部署示例
在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.3以上(x.x.x>1.0.3), RKNN版本在1.4.1b22以上。

```
# 下载图片和字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 下载部署示例代码
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/ocr/PP-OCR/rockchip/python

# 如果您希望从PaddleOCR下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到dygraph分支
git checkout dygraph
cd PaddleOCR/deploy/fastdeploy/rockchip/python


# CPU推理
python3 infer.py \
                --det_model ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                --cls_model ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                --rec_model ./ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                --rec_label_file ./ppocr_keys_v1.txt \
                --image 12.jpg \
                --device cpu

# NPU推理
python3 infer.py \
                --det_model ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer_rk3588_unquantized.rknn \
                --cls_model ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v20_cls_infer_rk3588_unquantized.rknn \
                --rec_model ./ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer_rk3588_unquantized.rknn \
                --rec_label_file ppocr_keys_v1.txt \
                --image 12.jpg \
                --device npu
```

运行完成可视化结果如下图所示
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">

## 4. 更多指南
- [PP-OCR系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/ocr.html)
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PP-OCRv3 C++部署](../cpp)
- [FastDeploy RKNPU2资源导航](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rknpu2.md)
- 如果用户想要调整前后处理超参数、单独使用文字检测识别模型、使用其他模型等，更多详细文档与说明请参考[PP-OCR系列在CPU/GPU上的部署](../../cpu-gpu/python/README.md)
