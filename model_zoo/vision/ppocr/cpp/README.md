# 编译PPOCR-DBDetector示例

当前支持模型版本为：[ch_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/models_list.md#1.1)

```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 下载模型和图片
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar
wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/doc/imgs/12.jpg

# 执行
./dbdetector_demo
```
执行完后可视化的结果保存在本地`vis.jpg`，同时会将检测框输出在终端，如下所示
```
det boxes: [[71,549],[431,539],[432,575],[72,585]]
det boxes: [[17,504],[518,482],[521,533],[20,554]]
det boxes: [[184,454],[401,445],[402,482],[185,491]]
det boxes: [[36,409],[487,385],[490,432],[39,456]]
Ocr Detect Done! Saved: vis.jpeg
```
