Paddle2ONNX支持将飞桨模型转换为ONNX格式存储。

由于不同框架的差异，如您在转换过程中出现错误，可随时通过issue向我们反馈，我们的工程师会及时在线回复[ISSUE](https://github.com/PaddlePaddle/paddle-onnx/issues/new)。


## 图像分类

目前已支持PaddlClas中大部分模型  [release/2.1](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1).

|Models | Source |  
|---|---|
| ResNet series| [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#ResNet_and_Vd_series)|
| Mobile series | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#Mobile_series)|
| SEResNeXt and Res2Net series | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#SEResNeXt_and_Res2Net_series)|
| DPN and DenseNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#DPN_and_DenseNet_series)|
| HRNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#HRNet_series)|
| Inception series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#Inception_series)|
| EfficientNet and ResNeXt101_wsl series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#EfficientNet_and_ResNeXt101_wsl_series)|
| ResNeSt and RegNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1#ResNeSt_and_RegNet_series)|


## OCR
支持PaddleOCR的轻量级和服务端文字识别模型 PaddleOCR [release/2.1](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1)。

| Models | Source |
|-------|--------|
|Chinese and English ultra-lightweight OCR model (9.4M) |[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15) |
|Chinese and English general OCR model (143.4M)|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15) |

## 语义分割
支持语义分割模型库PaddleSeg中的大部分模型 [release/v2.1](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1)。

| Models | Source |
|-------|--------|
|BiSeNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1/configs/bisenet) |
|DANet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/danet) |
|DeepLabv3|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/deeplabv3) |
|Deeplabv3P |[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/deeplabv3p) |
|FCN|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/fcn) |
|GCNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/gcnet) |
|OCRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/ocrnet) |
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/configs/unet) |

## 目标检测
支持目标检测模型库中8种检测结构 [release/2.1](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1)
| Models      | Source                                                       |
| ----------- | ------------------------------------------------------------ |
| YOLO-V3     | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/ |
| PPYOLO      | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/ |
| PPYOLO-Tiny | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/ |
| PPYOLO-V2   | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/ |
| TTFNet      | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ttfnet/ |
| PAFNet      | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ttfnet/ |
| SSD         | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ssd/ |
