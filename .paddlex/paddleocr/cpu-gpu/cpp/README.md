# PaddleOCR CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PP-OCRv3在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例.

PaddleOCR支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署OCR模型.

## 1. 运行部署示例

```bash  
# 找到部署包内的模型，这里测试的模型包括检测模型、分类模型和识别模型，如果只导出了其中的某个模型，则其他模型可用预训练模型进行测试
# 例如只导出了检测模型，则分类和识别模型可用预训练模型ch_ppocr_mobile_v2.0_cls_infer.tar和识别模型ch_PP-OCRv3_rec_infer.tar

# 可用于测试的预训练模型，可替换为自己训练的模型
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar

# 下载预测图片与字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 运行部署示例
# 在CPU上使用Paddle Inference推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# 在CPU上使用OenVINO推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
# 在CPU上使用ONNX Runtime推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2
# 在CPU上使用Paddle Lite推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 3
# 在GPU上使用Paddle Inference推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 4
# 在GPU上使用Paddle TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 5
# 在GPU上使用ONNX Runtime推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 6
# 在GPU上使用Nvidia TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 7

# 同时, FastDeploy提供文字检测,文字分类,文字识别三个模型的单独推理,
# 有需要的用户, 请准备合适的图片, 同时根据自己的需求, 参考infer.cc来配置自定义硬件与推理后端.

# 在CPU上,单独使用文字检测模型部署
./infer_det ./ch_PP-OCRv3_det_infer ./12.jpg 0

# 在CPU上,单独使用文字方向分类模型部署
./infer_cls ./ch_ppocr_mobile_v2.0_cls_infer ./12.jpg 0

# 在CPU上,单独使用文字识别模型部署
./infer_rec ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
```

## 2. 部署示例选项说明
在我们使用`infer_demo`时, 输入了6个参数, 分别为文字检测模型, 文字分类模型, 文字识别模型, 预测图片, 字典文件与最后一位的数字选项.
现在下表将解释最后一位数字选项的含义.
|数字选项|含义|
|:---:|:---:|
|0| 在CPU上使用Paddle Inference推理 |
|1| 在CPU上使用OenVINO推理 |
|2| 在CPU上使用ONNX Runtime推理 |
|3| 在CPU上使用Paddle Lite推理 |
|4| 在GPU上使用Paddle Inference推理 |
|5| 在GPU上使用Paddle TensorRT推理 |
|6| 在GPU上使用ONNX Runtime推理 |
|7| 在GPU上使用Nvidia TensorRT推理 |

## 3. 更多指南

### 3.1 如何使用C++部署PP-OCRv2系列模型.
本目录下的`infer.cc`代码是以PP-OCRv3模型为例, 如果用户有使用PP-OCRv2的需求, 只需要按照下面所示的方式, 来创建PP-OCRv2并使用.

```cpp
// 此行为创建PP-OCRv3模型的代码
auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);
// 只需要将PPOCRv3改为PPOCRv2,即可创造PPOCRv2模型, 同时, 后续的接口均使用ppocr_v2来调用
auto ppocr_v2 = fastdeploy::pipeline::PPOCRv2(&det_model, &cls_model, &rec_model);

// 如果用户在部署PP-OCRv2时, 需要使用TensorRT推理, 还需要改动Rec模型的TensorRT的输入shape.
// 建议如下修改, 需要把 H 维度改为32, W 纬度按需修改.
rec_option.SetTrtInputShape("x", {1, 3, 32, 10}, {rec_batch_size, 3, 32, 320},
                                {rec_batch_size, 3, 32, 2304});
```
### 3.2 如何在PP-OCRv2/v3系列模型中, 关闭文字方向分类器的使用.

在PP-OCRv3/v2中, 文字方向分类器是可选的, 用户可以按照以下方式, 来决定自己是否使用方向分类器.
```cpp
// 使用 Cls 模型
auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

// 不使用 Cls 模型
auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);

// 当不使用Cls模型时, 请删掉或者注释掉相关代码
```

### 3.3 如何修改前后处理超参数.
在示例代码中, 我们展示出了修改前后处理超参数的接口,并设置为默认值,其中, FastDeploy提供的超参数的含义与文档[PaddleOCR推理模型参数解释](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/inference_args.md)是相同的. 如果用户想要进行更多定制化的开发, 请阅读[PP-OCR系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1ocr.html)

```cpp
// 设置检测模型的max_side_len
det_model.GetPreprocessor().SetMaxSideLen(960);
// 其他...
```
