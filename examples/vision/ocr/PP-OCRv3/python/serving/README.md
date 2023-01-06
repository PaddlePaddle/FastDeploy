English | [简体中文](README_CN.md)

# PP-OCRv3 Python Simple Serving Demo

## Environment

- 1. Prepare environment and install FastDeploy Python whl, refer to [download_prebuilt_libraries](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. For installing the FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

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

# Launch server, change the configurations in server.py to select hardware, backend, etc.
# and use --host, --port to specify IP and port
fastdeploy simple_serving --app server:app
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
