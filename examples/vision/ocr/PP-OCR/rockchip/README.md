[English](README.md) | 简体中文

# PaddleOCR 模型在RKNPU2上部署方案-FastDeploy

## 1. 说明  
PaddleOCR支持通过FastDeploy在RKNPU2上部署相关模型.

## 2. 支持模型列表

下表中的模型下载链接由PaddleOCR模型库提供, 详见[PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)

| PaddleOCR版本 | 文本框检测 | 方向分类模型 | 文字识别 |字典文件| 说明 |
|:----|:----|:----|:----|:----|:--------|
| ch_PP-OCRv3[推荐] |[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv3系列原始超轻量模型，支持中英文、多语种文本检测 |
| en_PP-OCRv3[推荐] |[en_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [en_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) | [en_dict.txt](https://bj.bcebos.com/paddlehub/fastdeploy/en_dict.txt) | OCRv3系列原始超轻量模型，支持英文与数字识别，除检测模型和识别模型的训练数据与中文模型不同以外，无其他区别 |
| ch_PP-OCRv2 |[ch_PP-OCRv2_det](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv2_rec](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测 |
| ch_PP-OCRv2_mobile |[ch_ppocr_mobile_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测,比PPOCRv2更加轻量 |
| ch_PP-OCRv2_server |[ch_ppocr_server_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_server_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) |[ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2服务器系列模型, 支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|

## 3. 模型转换demo
```bash
# 下载PP-OCRv3文字检测模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar
# 下载文字方向分类器模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar
# 下载PP-OCRv3文字识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

# 请用户自行安装最新发布版本的paddle2onnx, 转换模型到ONNX格式的模型
paddle2onnx --model_dir ch_PP-OCRv3_det_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
            --enable_dev_version True
paddle2onnx --model_dir ch_ppocr_mobile_v2.0_cls_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
            --enable_dev_version True
paddle2onnx --model_dir ch_PP-OCRv3_rec_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
            --enable_dev_version True

# 固定模型的输入shape
python -m paddle2onnx.optimize --input_model ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                               --output_model ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                               --input_shape_dict "{'x':[1,3,960,960]}"
python -m paddle2onnx.optimize --input_model ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                               --output_model ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                               --input_shape_dict "{'x':[1,3,48,192]}"
python -m paddle2onnx.optimize --input_model ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                               --output_model ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                               --input_shape_dict "{'x':[1,3,48,320]}"

# 在rockchip/rknpu2_tools/目录下, 我们为用户提供了转换ONNX模型到RKNN模型的工具
python tools/rknpu2/export.py --config_path tools/rknpu2/config/ppocrv3_det.yaml \
                              --target_platform rk3588
python tools/rknpu2/export.py --config_path tools/rknpu2/config/ppocrv3_rec.yaml \
                              --target_platform rk3588
python tools/rknpu2/export.py --config_path tools/rknpu2/config/ppocrv3_cls.yaml \
                              --target_platform rk3588
```

## 3. 详细部署的部署示例  
- [Python部署](python)
- [C++部署](cpp)
