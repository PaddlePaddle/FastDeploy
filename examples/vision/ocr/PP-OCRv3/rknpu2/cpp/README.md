[English](README.md) | 简体中文
# PPOCRv3 C++部署示例

本目录下提供`infer.cc`快速完成PPOCRv3在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认你已经成功完成以下两个操作:

* [正确编译FastDeploy SDK](../../../../../../docs/cn/faq/rknpu2/build.md).
* [成功转换模型](../README.md).

在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.3以上(x.x.x>1.0.3), RKNN版本在1.4.1b22以上。

```
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载图片和字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt


# 拷贝RKNN模型到build目录

# CPU推理
./infer_static_shape_demo ./ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                          ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                          ./ch_PP-OCRv3_rec_infer\ch_PP-OCRv3_rec_infer.onnx \
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


运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">

## 其它文档

- [C++ API查阅](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/)
- [PPOCR 系列模型介绍](../../../README_CN.md)
- [PPOCRv3 Python部署](../python)
- [模型预测结果说明](../../../../../../docs/cn/faq/how_to_change_backend.md)
