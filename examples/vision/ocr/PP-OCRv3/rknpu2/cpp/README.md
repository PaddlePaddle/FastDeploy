English | [简体中文](README_CN.md)
# PPOCRv3 C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of PPOCRv3 on CPU/GPU and GPU accelerated by TensorRT.

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j


# Download model, image, and dictionary files
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU推理
./infer_static_shape_demo ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                          ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                          ./ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                          ./ppocr_keys_v1.txt \
                          ./12.jpg \
                          0
# RKNPU推理
./infer_static_shape_demo ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer_rk3588_unquantized.rknn \
                            ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v20_cls_infer_rk3588_unquantized.rknn \
                             ./ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer_rk3588_unquantized.rknn \
                              ./ppocr_keys_v1.txt \
                              ./12.jpg \
                              1
```

The above command works for Linux or MacOS. For SDK in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../../docs/cn/faq/use_sdk_on_windows.md)

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">

## Other Documents

- [C++ API Reference](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/)
- [PPOCR Model Description](../README.md)
- [PPOCRv3 Python Deployment](../python)
- [Model Prediction Results](../../../../../../docs/en/faq/how_to_change_backend.md)
- [How to switch the model inference backend engine](../../../../../../docs/en/faq/how_to_change_backend.md)
