# PaddleOCR 模型部署

## PaddleOCR为多个模型组合串联任务，包含
- 文本检测 `DBDetector`
- [可选]方向分类 `Classifer` 用于调整进入文字识别前的图像方向
- 文字识别 `Recognizer` 用于从图像中识别出文字

根据不同场景, FastDeploy汇总提供如下OCR任务部署, 用户需同时下载3个模型（或2个，分类器可选), 完成OCR整个预测流程

### 高精度OCR识别

如下模型识别精度高，适合GPU及高性能CPU场景下的部署

| OCR版本 | 文本框检测 | 方向分类模型 | 文字识别 | 说明 |
|:------- |:------ |:-------  | :-------- | :--- |
| PPOCRv2 |[ch_PP-OCRv2_det](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_PP-OCRv2_rec](https://bj.bcebos.com/paddlehub/fastdeploy/ch_PP-OCRv2_rec_infer) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测 |
| PPOCRv2_Server |[ch_ppocr_server_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) |  | [ch_ppocr_server_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) | OCRv2服务器系列模型, 支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|
| PPOCRv3 |[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) | OCRv3系列原始超轻量模型，支持中英文、多语种文本检测 |


### 高性能OCR识别

如下是轻量级模型，识别速度快，适合要求推理速度快或者低算力场景下的部署

| OCR版本 | 文本框检测 | 方向分类模型 | 文字识别 | 说明 |
|:------- |:------ |:-------  | :-------- | :--- |
| PPOCRv2_mobile |[ch_ppocr_mobile_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_ppocr_mobile_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测 |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
