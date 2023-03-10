# OCR模型库

本文档中模型库均来源于PaddleOCR [release/2.4分支](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4)，在下表中提供了部分已经转换好的模型，如有更多模型或自行模型训练导出需求，可参考 [PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/models_list.md).

| 模型简介                              | 模型名称                | 推荐场景        | 检测模型                                                     | 方向分类器                                                   | 识别模型                                                     |
| ------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中英文超轻量PP-OCRv2模型（13.0M）     | ch_PP-OCRv2_xx          | 移动端 | [Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_det_infer.tar) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_det_infer.onnx) | [Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/ch_ppocr_mobile_v2.0_cls_infer.tar) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/ch_ppocr_mobile_v2.0_cls_infer.onnx) | [Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_rec_infer.tar) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_rec_infer.onnx) |



# ONNX模型推理示例

各模型的推理前后处理参考本目录下的infer.py，以中英文超轻量PP-OCRv2模型为例，如下命令即可得到推理结果

```bash
# 安装onnxruntime
pip3 install onnxruntime

# 下载det模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_det_infer.onnx

# 下载rec模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_rec_infer.onnx

# 下载cls模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/ch_ppocr_mobile_v2.0_cls_infer.onnx

python3 infer.py  \
--det_model_dir=./ch_PP-OCRv2_det_infer.onnx  \
--rec_model_dir=./ch_PP-OCRv2_rec_infer.onnx  \
--cls_model_dir=./ch_ppocr_mobile_v2.0_cls_infer.onnx  \
--image_path=./images/lite_demo.png
```

你也可以使用Paddle框架进行推理验证

```bash
wget -nc  -P ./inference https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_det_infer.tar
cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && cd ..

wget -nc  -P ./inference https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_rec_infer.tar
cd ./inference && tar xf ch_PP-OCRv2_rec_infer.tar && cd ..

wget -nc  -P ./inference https://bj.bcebos.com/paddle2onnx/model_zoo/ch_ppocr_mobile_v2.0_cls_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..

python3 infer.py \
--cls_model_dir=./inference/ch_ppocr_mobile_v2.0_cls_infer \
--rec_model_dir=./inference/ch_PP-OCRv2_rec_infer \
--det_model_dir=./inference/ch_PP-OCRv2_det_infer \
--image_path=./images/lite_demo.png \
--use_paddle_predict=True
```

最后ONNXRuntime和Paddle终端输出结果，都是如下：

```
The, 0.984
visualized, 0.882
etect18片, 0.720
image saved in./vis.jpg, 0.947
纯臻营养护发素0.993604, 0.996
产品信息/参数, 0.922
0.992728, 0.914
（45元／每公斤，100公斤起订）, 0.926
0.97417, 0.977
每瓶22元，1000瓶起订）0.993976, 0.962
【品牌】：代加工方式/0EMODM, 0.945
0.985133, 0.980
【品名】：纯臻营养护发素, 0.921
0.995007, 0.883
【产品编号】：YM-X-30110.96899, 0.955
【净含量】：220ml, 0.943
Q.996577, 0.932
【适用人群】：适合所有肤质, 0.913
0.995842, 0.969
【主要成分】：鲸蜡硬脂醇、燕麦B-葡聚, 0.883
0.961928, 0.964
10, 0.812
糖、椰油酰胺丙基甜菜碱、泛醒, 0.866
0.925898, 0.943
（成品包材）, 0.974
心, 0.691
0.972573, 0.961
【主要功能】：可紧致头发磷层，从而达到, 0.936
0.994448, 0.952
13, 0.998
即时持久改善头发光泽的效果，给干燥的头, 0.994
0.990198, 0.975
14, 0.977
发足够的滋养, 0.991
0.997668, 0.918
花费了0.457335秒, 0.901
Finish!
```
