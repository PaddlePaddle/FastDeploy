简体中文 | [English](README_EN.md)

# PP-OCRv3 Python轻量服务化部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

服务端：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/ocr/PP-OCRv3/python/serving

# 下载模型和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 安装uvicorn
pip install uvicorn

# 启动服务，可选择是否使用GPU和TensorRT，可根据uvicorn --help配置IP、端口号等
# CPU
DET_MODEL_DIR=ch_PP-OCRv3_det_infer CLS_MODEL_DIR=ch_ppocr_mobile_v2.0_cls_infer REC_MODEL_DIR=ch_PP-OCRv3_rec_infer REC_LABEL_FILE=ppocr_keys_v1.txt DEVICE=cpu uvicorn server:app
# GPU
DET_MODEL_DIR=ch_PP-OCRv3_det_infer CLS_MODEL_DIR=ch_ppocr_mobile_v2.0_cls_infer REC_MODEL_DIR=ch_PP-OCRv3_rec_infer REC_LABEL_FILE=ppocr_keys_v1.txt DEVICE=gpu uvicorn server:app
# GPU上使用TensorRT （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
DET_MODEL_DIR=ch_PP-OCRv3_det_infer CLS_MODEL_DIR=ch_ppocr_mobile_v2.0_cls_infer REC_MODEL_DIR=ch_PP-OCRv3_rec_infer REC_LABEL_FILE=ppocr_keys_v1.txt DEVICE=gpu USE_TRT=true uvicorn server:app
```

客户端：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/ocr/PP-OCRv3/python/serving

# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

# 请求服务，获取推理结果（如有必要，请修改脚本中的IP和端口号）
python client.py
```
