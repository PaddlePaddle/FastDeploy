[简体中文](../cn/quantize.md) ｜English 

# Quantization Acceleration

Quantization is a popular method in model compression to obtain models with smaller sizes and faster inference speeds.
Based on PaddleSlim, FastDeploy integrates a one-click model quantization tool. FastDeploy supports the deployment of quantized models for faster inference.

## FastDeploy supports quantitative model deployment for multiple engines and hardware

Currently, multiple inference backends in FastDeploy can deploy quantitative models on different hardware. 

| Hardware / Inference Backend | ONNX Runtime | Paddle Inference | TensorRT |
|:---------------------------- |:------------ |:---------------- |:-------- |
| CPU                          | Support      | Support          |          |
| GPU                          |              |                  | Support  |

## Model Quantization

### Quantization Methods

Based on PaddleSlim, FastDeploy currently offers knowledge distillation training and post-training quantization. Knowledge distillation training quantizes models by model training, and post-training quantization can complete the model quantization without training. FastDeploy can deploy quantization models from both approaches.

The comparison between both methods is as follows.

| Method                          | Time                                            | Accuracy                              | Size     | Inference Speed |
| ------------------------------- | ----------------------------------------------- | ------------------------------------- | -------- | --------------- |
| Post-training quantization      | No training needed, fast deployment             | Slightly lower than distillation      | The same | The same        |
| Knowledge distillation training | Training needed, and takes slightly longer time | Slight lower than un-quantized models | The same | The same        |

### Use FastDeploy One-Click Quantization Tool

Fastdeploy, based on PaddleSlim, provides users with a one-click model quantification tool. Please refer to the following document for model quantification.

- [FastDeploy One-Click Quantization Tool](../../tools/quantization/)
  Once developers obtain a quantitative model, they can use FastDeploy to deploy the quantized model.

## Quantization Demo

Now, FastDeploy supports:

### YOLO  Series

| Model                                                       | Inference Backend | Hardware | FP32 Inference Time Delay/ms | INT8  Inference Time Delay/ms | Acceleration Ratio | FP32 mAP | INT8 mAP | Method                 |
| ----------------------------------------------------------- | ----------------- | -------- | ---------------------------- | ----------------------------- | ------------------ | -------- | -------- | ---------------------- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT         |    GPU    |  14.13        |  11.22      |      1.26         | 37.6  | 36.6 | Knowledge Distillation training |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime     |    CPU    |  183.68       |    100.39   |      1.83         | 37.6  | 33.1 | Knowledge Distillation training |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference  |    CPU    |      226.36   |   152.27     |      1.48         |37.6 | 36.8 | Knowledge Distillation training |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT         |    GPU    |       12.89        |   8.92          |  1.45             | 42.5 | 40.6| Knowledge Distillation training |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | ONNX Runtime     |    CPU    |   345.85            |  131.81           |      2.60         |42.5| 36.1| Knowledge Distillation training |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |         366.41      |    131.70         |     2.78          |42.5| 41.2| Knowledge Distillation training |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT          |    GPU    |     30.43          |      15.40       |       1.98        | 51.1| 50.8| Knowledge Distillation training |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     971.27          |  471.88           |  2.06             | 51.1 | 42.5| Knowledge Distillation training |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |          1015.70     |      562.41       |    1.82           |51.1 | 46.3| Knowledge Distillation training |


The data in the above table shows the end-to-end inference performance of FastDeploy deployment **before and after **model quantization.

- The test data are images from the COCO2017 validation set.
- The inference time delay is the inference latency on different Runtimes in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version 8.4.15, and the number of fixed CPU threads is 1 in all tests.

### PaddleClas Series

| Model                                                                         | Inference Backend | Hardware | FP32 Inference Time Delay/ms | INT8 nference Time Delay/ms | Acceleration Ratio | FP32 Top1 | INT8 Top1 | MEthod                     |
| ----------------------------------------------------------------------------- | ----------------- | -------- | ---------------------------- | --------------------------- | ------------------ | --------- | --------- | -------------------------- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)      | ONNX Runtime      | CPU      | 77.20                        | 40.08                       | 1.93               | 79.12     | 78.87     | Post-training quantization |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)      | TensorRT          | GPU      | 3.70                         | 1.80                        | 2.06               | 79.12     | 79.06     | Post-training quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/) | ONNX Runtime      | CPU      | 30.99                        | 10.24                       | 3.03               | 77.89     | 75.09     | Post-training quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/) | TensorRT          | GPU      | 1.80                         | 0.58                        | 3.10               | 77.89     | 76.86     | Post-training quantization |



The data in the above table shows the end-to-end inference performance of FastDeploy deployment **before and after** model quantization.

- The test data are images from the ImageNet-2012 validation set.
- The inference latency is the inference latency in milliseconds (ms) on different Runtimes.
- The CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version 8.4.15, and the number of fixed CPU threads is 1 for all tests.
