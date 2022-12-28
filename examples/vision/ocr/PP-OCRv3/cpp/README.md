# PPOCRv3 C++部署示例

本目录下提供`infer.cc`快速完成PPOCRv3在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

```
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
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
# GPU上TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2
# GPU上Paddle-TRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 3
# 昆仑芯XPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 4
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">

## 其它文档

- [C++ API查阅](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/)
- [PPOCR 系列模型介绍](../../)
- [PPOCRv3 Python部署](../python)
- [模型预测结果说明](../../../../../docs/cn/faq/how_to_change_backend.md)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
