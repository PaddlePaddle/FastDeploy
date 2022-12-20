English | [简体中文](../cn/quantize.md) 

# Quantize Model for Acceleration
Quantization is a popular method of model compression, and quantized models have smaller size and faster inference speed.
Based on PaddleSlim, FastDeploy integrates an one-click model quantization tool, and supports to deploy quantized models to help achieve inference acceleration.



## Multiple engines and hardware supporting quantized model deployment
Currently multiple inference backends in FastDeploy can support the deployment of quantized models on different hardware as follows:

| Hardware/Inference backends | ONNX Runtime | Paddle Inference | TensorRT |
| :-----------| :--------   | :--------------- | :------- |
|   CPU       |  support        |  support            |          |  
|   GPU       |             |                  | support      |


## model quantization

### Quantization Methods
Based on PaddleSlim, currently the quantization methods provided by FastDeploy are distillation training quantization and offline quantization. Distillation training quantization obtains the quantized model by training models, while offline quantization does not require model training. FastDeploy can deploy quantized models from both approaches.

The main differences between the two approaches are shown in the following table:
| Quantization Method | Time Consuming | Model Accuracy | Model Size | Inference Speed |
| :-----------| :--------| :-------| :------- | :------- |
|   Offline quantization      |  Short(no training required)|  Lower accuracy       | Both are the same   | Both are the same   |  
|   Distillation training quantization      | Long(requires training) |  More loss are produced | Both are the same   |Both are the same   |  

### Using One-click Model Quantization tool to quantize model
Based on PaddleSlim, Fastdeploy provides users with an one-click model quantification tool, please refer to the following document for model quantification:
- [An One-click Model Quantization tool](../../tools/auto_compression/)
Once you obtain a quantized model output, you can use FastDeploy to deploy it.


## Quantification example
The supported quantification model in FastDeploy is shown in the table below:

### YOLO Series
| Model                 |Inference Backend            |Deployment Hardware    | Inference Time Delay in FP32    | Inference Time Delay in INT8  | Acceleration Ratio    | FP32 mAP | INT8 mAP | Quantization Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT         |    GPU    |  8.79       |  5.17     |      1.70         | 37.6  | 36.6 |  Distillation training quantization |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime     |    CPU    |  176.34      |    92.95   |      1.90        | 37.6  | 33.1 | Distillation training quantization |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference  |    CPU    |      217.05  |   133.31     |     1.63         |37.6 | 36.8 |  Distillation training quantization |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT         |    GPU    |       8.60       |   5.16         |  1.67            | 42.5 | 40.6| Distillation training quantization |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | ONNX Runtime     |    CPU    |   338.60           |  128.58          |      2.60         |42.5| 36.1| Distillation training quantization |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |        356.62     |    125.72        |     2.84         |42.5| 41.2| Distillation training quantization |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT          |    GPU    |     24.57         |      9.40     |      2.61       | 51.1| 50.8| Distillation training quantization |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     976.88         |  462.69          |  2.11            | 51.1 | 42.5| Distillation training quantization |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |         1022.55    |     490.87      |   2.08         |51.1 | 46.3| Distillation training quantization |

The data in the table above are the Runtime inference performance in FastDeploy before and after model quantification.
- The test data are images in the COCO2017 validation set.
- The inference time delay represents the time delay in different Runtime, in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version is 8.4.15, and the number of fixed CPU threads is 1.



### PaddleDetection Series
| Model                 |Inference Backend            |Deployment Hardware    | Inference Time Delay in FP32    | Inference Time Delay in INT8  | Acceleration Ratio    | FP32 mAP | INT8 mAP |Quantization Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | TensorRT         |    GPU    |  24.52       |  11.53    |      2.13        | 51.4  | 50.7 | Distillation training quantization |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | ONNX Runtime |    CPU    |     1085.62 |   457.56     |      2.37        |51.4 | 50.0 |Distillation training quantization |

The data in the table above are the Runtime inference performance in FastDeploy before and after model quantification.
- The test data are images in the COCO2017 validation set.
- The inference time delay represents the time delay in different Runtime, in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version is 8.4.15, and the number of fixed CPU threads is 1.



### PaddleClas系列
| Model                 |Inference Backend            |Deployment Hardware    | Inference Time Delay in FP32    | Inference Time Delay in INT8  | Acceleration Ratio    | FP32 Top1 | INT8 Top1 |Quantization Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime         |    CPU    |  77.20       |  40.08     |     1.93        | 79.12  | 78.87|  Offline quantization|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  3.70        | 1.80      |      2.06      | 79.12  | 79.06 | Offline quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)             | ONNX Runtime |    CPU    |     30.99   |   10.24    |     3.03        |77.89 | 75.09 |Offline quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)             | TensorRT  |    GPU    |     1.80  |   0.58    |      3.10       |77.89 | 76.86 | Offline quantization |

The data in the table above are the Runtime inference performance in FastDeploy before and after model quantification.
- The test data are images in the ImageNet-2012 validation set.
- The inference time delay represents the time delay in different Runtime, in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version is 8.4.15, and the number of fixed CPU threads is 1.

