English | [简体中文](README_CN.md)

# PP-OCRv3 Python Simple Serving Demo

## Environment

- 1. Prepare environment and install FastDeploy Python whl, refer to [download_prebuilt_libraries](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Server:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/ocr/PP-OCRv3/python/serving

# Download models and labels
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# Launch server, it's configurable to use GPU and TensorRT,
# and use --host, --port to specify IP and port, etc.
# CPU
DET_MODEL_DIR=ch_PP-OCRv3_det_infer CLS_MODEL_DIR=ch_ppocr_mobile_v2.0_cls_infer REC_MODEL_DIR=ch_PP-OCRv3_rec_infer REC_LABEL_FILE=ppocr_keys_v1.txt DEVICE=cpu fastdeploy simple_serving --app server:app
# GPU
DET_MODEL_DIR=ch_PP-OCRv3_det_infer CLS_MODEL_DIR=ch_ppocr_mobile_v2.0_cls_infer REC_MODEL_DIR=ch_PP-OCRv3_rec_infer REC_LABEL_FILE=ppocr_keys_v1.txt DEVICE=gpu fastdeploy simple_serving --app server:app
# GPU and TensorRT
DET_MODEL_DIR=ch_PP-OCRv3_det_infer CLS_MODEL_DIR=ch_ppocr_mobile_v2.0_cls_infer REC_MODEL_DIR=ch_PP-OCRv3_rec_infer REC_LABEL_FILE=ppocr_keys_v1.txt DEVICE=gpu BACKEND=trt fastdeploy simple_serving --app server:app
```

Client:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/ocr/PP-OCRv3/python/serving

# Download test image
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

# Send request and get inference result (Please adapt the IP and port if necessary)
python client.py
```
