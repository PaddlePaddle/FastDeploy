# PaddleOCR 模型部署

## PaddleOCR为多个模型组合串联任务，包含
- 文本检测 `DBDetector`
- [可选]方向分类 `Classifer` 用于调整进入文字识别前的图像方向
- 文字识别 `Recognizer` 用于从图像中识别出文字

根据不同场景, FastDeploy汇总提供如下OCR任务部署, 用户需同时下载3个模型与字典文件（或2个，分类器可选), 完成OCR整个预测流程

### OCR 中英文系列模型

| OCR版本 | 文本框检测 | 方向分类模型 | 文字识别 |字典文件| 说明 |
|:----|:----|:----|:----|:----|:--------|
| PPOCRv3[推荐] |[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv3系列原始超轻量模型，支持中英文、多语种文本检测 |
| PPOCRv3[推荐] |[en_PP-OCRv3_det](https://bj.bcebos.com/paddlehub/fastdeploy/en_PP-OCRv3_det_infer.tar.gz) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [en_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) | [en_dict.txt](https://bj.bcebos.com/paddlehub/fastdeploy/en_dict.txt) | OCRv3系列原始超轻量模型，支持英文与数字识别，除检测模型和识别模型的训练数据与中文模型不同以外，无其他区别 |
| PPOCRv2 |[ch_PP-OCRv2_det](https://bj.bcebos.com/paddlehub/fastdeploy/ch_PP-OCRv2_det_infer.tar.gz) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_PP-OCRv2_rec](https://bj.bcebos.com/paddlehub/fastdeploy/ch_PP-OCRv2_rec_infer.tar.gz) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测 |
| PPOCRv2_mobile |[ch_ppocr_mobile_v2.0_det](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_det_infer.tar.gz) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_ppocr_mobile_v2.0_rec](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_rec_infer.tar.gz) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测,比PPOCRv2更加轻量 |
| PPOCRv2_server |[ch_ppocr_server_v2.0_det](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_server_v2.0_det_infer.tar.gz) | [ch_ppocr_mobile_v2.0_cls](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz) | [ch_ppocr_server_v2.0_rec](https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_server_v2.0_rec_infer.tar.gz) |[ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2服务器系列模型, 支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|

### OCR 模型的处理说明

为了让OCR系列模型在FastDeploy多个推理后端上正确推理，以上表格中的部分模型的输入shape，和PaddleOCR套件提供的模型有差异.
例如，由PaddleOCR套件库提供的英文版PP-OCRv3_det模型,输入的shape是`[-1,3,960,960]`, 而FastDeploy提供的此模型输入shape为`[-1,3,-1,-1]`.
所以用户在FastDeploy上推理PaddleOCR提供的模型，可能会存在shape上的报错.我们推荐用户直接下载FastDeploy提供的模型, 用户也可以参考如下工具仓库，自行修改模型的输入shape.

仓库链接: https://github.com/jiangjiajun/PaddleUtils

使用示例如下：
```
#该用例将en_PP-OCRv3_det_infer模型的输入shape, 改为[-1,3,-1,-1], 并将新模型存放至output文件夹下
git clone git@github.com:jiangjiajun/PaddleUtils.git
cd paddle
python paddle_infer_shape.py --model_dir en_PP-OCRv3_det_infer/ \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir output  \
                             --input_shape_dict="{'x':[-1,3,-1,-1]}"
```
