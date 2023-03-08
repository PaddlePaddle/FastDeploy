English | [简体中文](README_CN.md)
# PPOCRv3 Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2.  Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of PPOCRv3 on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/ocr/PP-OCRv3/python/

python3 infer_static_shape.py \
                --det_model ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                --cls_model ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                --rec_model ./ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                --rec_label_file ./ppocr_keys_v1.txt \
                --image 12.jpg \
                --device cpu

# NPU推理
python3 infer_static_shape.py \
                --det_model ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer_rk3588_unquantized.rknn \
                --cls_model ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v20_cls_infer_rk3588_unquantized.rknn \
                --rec_model ./ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer_rk3588_unquantized.rknn \
                --rec_label_file ppocr_keys_v1.txt \
                --image 12.jpg \
                --device npu
```

The visualized result after running is as follows
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">




## Other Documents

- [Python API reference](https://baidu-paddle.github.io/fastdeploy-api/python/html/)
- [PPOCR Model Description](../README.md)
- [PPOCRv3 C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../../docs/api/vision_results/README.md)
- [How to switch the model inference backend engine](../../../../../../docs/en/faq/how_to_change_backend.md)
