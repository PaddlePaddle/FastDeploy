[中文](../cn/quantize.md) | English

# Quantization Acceleration
Quantization is a popular method of model compression, resulting in smaller models size and faster inference speed.
Based on PaddleSlim's Auto Compression Toolkit (ACT), FastDeploy provides users with a one-click model compression automation tool. This tool includes a variety of strategies for auto-compression, the current main strategies are post-trainning quantization and quantaware distillation training. At the same time, FastDeploy supports the deployment of compressed models to help users achieve inference acceleration.

## Multiple inference engines and hardware support for quantized model deployment in FastDeploy
Currently, multiple inference engines in FastDeploy can support the deployment of quantized models on different hardware.

| Hardware/Inference engine | ONNX Runtime | Paddle Inference | TensorRT | Paddle-TensorRT |
| :-----------| :--------   | :--------------- | :------- | :------- |
|   CPU       |  Support        |  Support            |          |           |  
|   GPU       |             |                  | Support      |    Support        |

## Model Quantization

### Quantization Method
Based on PaddleSlim, the quantization methods currently provided by FastDeploy one-click model auto-compression are quantaware distillation training and post training quantization, quantaware distillation training to obtain quantization models through model training, and post training  quantization to complete the quantization of models without model training. FastDeploy can deploy the quantized models produced by both methods.

The comparison of the two methods is shown in the following table:
| Method | Time Cost | Quantized Model Accuracy | Quantized Model Size | Inference Speed |
| :-----------| :--------| :-------| :------- | :------- |
|   Post Training Quantization      |  Less than Quantware|  Lower than Quantaware   | Same   | Same   |  
|   Quantaware Distillation Training      | Normal  |  Lower than FP32 Model | Same   |Same   |  

### Use FastDeploy one-click model automation compression tool to quantify models
Based on PaddleSlim's Auto Compression Toolkit (ACT), FastDeploy provides users with a one-click model automation compression tool, please refer to the following document for one-click model automation compression.
- [FastDeploy One-Click Model Automation Compression](../../tools/common_tools/auto_compression/README_EN.md)

## Benchmark
Currently, FastDeploy supports automated compression, and the Runtime Benchmark and End-to-End Benchmark of the model that completes the deployment test are shown below.

NOTE:
- Runtime latency is the inference latency of the model on various Runtimes, including CPU->GPU data copy, GPU inference, and GPU->CPU data copy time. It does not include the respective pre and post processing time of the models.
- The end-to-end latency is the latency of the model in the actual inference scenario, including the pre and post processing of the model.
- The measured latencies are averaged over 1000 inferences, in milliseconds.
- INT8 + FP16 is to enable the FP16 inference option for Runtime while inferring the INT8 quantization model.
- INT8 + FP16 + PM is the option to use Pinned Memory while inferring INT8 quantization model and turning on FP16, which can speed up the GPU->CPU data copy speed.
- The maximum speedup ratio is obtained by dividing the FP32 latency by the fastest INT8 inference latency.
- The strategy is quantitative distillation training, using a small number of unlabeled data sets to train the quantitative model, and verify the accuracy on the full validation set, INT8 accuracy does not represent the highest INT8 accuracy.
- The CPU is Intel(R) Xeon(R) Gold 6271C with a fixed CPU thread count of 1 in all tests. The GPU is Tesla T4, TensorRT version 8.4.15.

### YOLO Series
#### Runtime Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  | Max Speedup    | FP32 mAP | INT8 mAP | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT   |    GPU    |  7.87    | 4.51 |  4.31     | 3.17     |      2.48         | 37.6  | 36.7 | Quantaware Distillation Training  |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | Paddle-TensorRT  |    GPU   |  7.99    |  None |  4.46    | 3.31     |      2.41         | 37.6  | 36.8 | Quantaware Distillation Training  |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime   |    CPU    |  176.41      |    91.90   |  None |  None |      1.90        | 37.6  | 33.1 |Quantaware Distillation Training  |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference|    CPU    |      213.73  |   130.19     |  None  | None |   1.64     |37.6 | 35.2 | Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT  |    GPU    |       9.47    |   3.23    |  4.09      |2.81    |  3.37            | 42.5 | 40.7|Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | Paddle-TensorRT |    GPU    |       9.31    | None|  4.17  | 2.95       |  3.16            | 42.5 | 40.7|Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)          | ONNX Runtime     |    CPU    |   334.65     |  126.38      | None | None|     2.65   |42.5| 36.8|Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |    352.87   |    123.12    |None | None|     2.87         |42.5| 40.8|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT   |    GPU    |     27.47    |  6.52   |  6.74| 5.19|    5.29       | 51.1| 50.4|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | Paddle-TensorRT |    GPU    |     27.87|None|6.91|5.86      |      4.76       | 51.1| 50.4|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     996.65        |  467.15 |None|None          |  2.13           | 51.1 | 43.3|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |     995.85  |     477.93|None|None      |   2.08         |51.1 | 46.2|Quantaware Distillation Training  |

#### End2End Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency  | INT8+FP16+PM End2End Latency  | Max Speedup    | FP32 mAP | INT8 mAP | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT   |    GPU    |  24.61   | 21.20 |  20.78     | 20.94     |      1.18         | 37.6  | 36.7 | Quantaware Distillation Training  |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | Paddle-TensorRT  |    GPU   |  23.53    |  None |  21.98    | 19.84     |      1.28        | 37.6  | 36.8 | Quantaware Distillation Training  |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime   |    CPU    |  197.323      |    110.99   |  None |  None |      1.78        | 37.6  | 33.1 |Quantaware Distillation Training  |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference|    CPU    |      235.73  |   144.82     |  None  | None |   1.63     |37.6 | 35.2 | Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT  |    GPU    |       15.66    |   11.30   |  10.25      |9.59   |  1.63           | 42.5 | 40.7|Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | Paddle-TensorRT |    GPU    |       15.03   | None|  11.36 | 9.32       |  1.61            | 42.5 | 40.7|Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)          | ONNX Runtime     |    CPU    |   348.21    |  126.38      | None | None| 2.82       |42.5| 36.8|Quantaware Distillation Training  |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |    352.87   |    121.64    |None | None|    3.04       |42.5| 40.8|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT   |    GPU    |     36.47   |  18.81  |  20.33| 17.58|    2.07      | 51.1| 50.4|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | Paddle-TensorRT |    GPU    |     37.06|None|20.26|17.53    |      2.11      | 51.1| 50.4|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     988.85       |  478.08 |None|None          |  2.07          | 51.1 | 43.3|Quantaware Distillation Training  |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |     1031.73 |     500.12|None|None      |   2.06         |51.1 | 46.2|Quantaware Distillation Training  |



### PaddleClasSeries
#### Runtime Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  | Max Speedup    | FP32 Top1 | INT8 Top1 | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  3.55 | 0.99|0.98|1.06  |      3.62      | 79.12  | 79.06 | Post Training Quantization |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle-TensorRT  |    GPU    |  3.46 |None |0.87|1.03  |      3.98      | 79.12  | 79.06 | Post Training Quantization |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime    |    CPU    |  76.14       |  35.43  |None|None  |     2.15        | 79.12  | 78.87|  Post Training Quantization|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle Inference  |    CPU    |  76.21       |  24.01 |None|None  |     3.17       | 79.12  | 78.55 |  Post Training Quantization|
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | TensorRT  |    GPU    |     0.91 |   0.43 |0.49 | 0.54    |      2.12       |77.89 | 76.86 | Post Training Quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | Paddle-TensorRT   |    GPU    |  0.88|   None| 0.49|0.51 |      1.80      |77.89 | 76.86 | Post Training Quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | ONNX Runtime |    CPU    |     30.53   |   9.59|None|None    |     3.18       |77.89 | 75.09 |Post Training Quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        |  Paddle Inference  |    CPU    |     12.29  |   4.68  |     None|None|2.62       |77.89 | 71.36 |Post Training Quantization |

#### End2End Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency  | INT8+FP16+PM End2End Latency  | Max Speedup    | FP32 Top1 | INT8 Top1 | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  4.92| 2.28|2.24|2.23 |      2.21     | 79.12  | 79.06 | Post Training Quantization |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle-TensorRT  |    GPU    |  4.48|None |2.09|2.10 |      2.14   | 79.12  | 79.06 | Post Training Quantization |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime    |    CPU    |  77.43    |  41.90 |None|None  |     1.85        | 79.12  | 78.87|  Post Training Quantization|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle Inference  |    CPU    |   80.60     |  27.75 |None|None  |     2.90     | 79.12  | 78.55 |  Post Training Quantization|
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | TensorRT  |    GPU    |     2.19 |   1.48|1.57| 1.57   |      1.48     |77.89 | 76.86 | Post Training Quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | Paddle-TensorRT   |    GPU    |  2.04|   None| 1.47|1.45 |      1.41     |77.89 | 76.86 | Post Training Quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | ONNX Runtime |    CPU    |     34.02  |   12.97|None|None    |    2.62       |77.89 | 75.09 |Post Training Quantization |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        |  Paddle Inference  |    CPU    |    16.31 |   7.42  |     None|None| 2.20      |77.89 | 71.36 |Post Training Quantization |



### PaddleDetectionSeries
#### Runtime Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  | Max Speedup    | FP32 mAP | INT8 mAP | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | TensorRT         |    GPU    |  27.90 | 6.39 |6.44|5.95    |      4.67       | 51.4  | 50.7 | Quantaware Distillation Training  |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | Paddle-TensorRT |    GPU    |  30.89     |None  |  13.78 |14.01    |      2.24       | 51.4  | 50.5| Quantaware Distillation Training  |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | ONNX Runtime |    CPU    |     1057.82 |   449.52 |None|None    |      2.35        |51.4 | 50.0 |Quantaware Distillation Training  |


#### End2End Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency  | INT8+FP16+PM End2End Latency  | Max Speedup    | FP32 mAP | INT8 mAP | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | TensorRT         |    GPU    |  35.75 | 15.42 |20.70|20.85  |      2.32      | 51.4  | 50.7 | Quantaware Distillation Training  |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | Paddle-TensorRT |    GPU    | 33.48    |None  |  18.47 |18.03   |     1.81       | 51.4  | 50.5| Quantaware Distillation Training  |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | ONNX Runtime |    CPU    |     1067.17 |   461.037 |None|None    |      2.31        |51.4 | 50.0 |Quantaware Distillation Training  |



### PaddleSegSeries
#### Runtime Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  | Max Speedup    | FP32 mIoU | INT8 mIoU | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](../../examples/vision/segmentation/paddleseg/quantize)  | Paddle Inference |    CPU    |     1138.04|   602.62 |None|None     |      1.89      |77.37 | 71.62 |Quantaware Distillation Training  |

#### End2End Benchmark
| Model                 |Inference Backends            |Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency  | INT8+FP16+PM End2End Latency  | Max Speedup    | FP32 mIoU | INT8 mIoU | Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](../../examples/vision/segmentation/paddleseg/quantize)  | Paddle Inference |    CPU    |     4726.65|   4134.91|None|None     |      1.14      |77.37 | 71.62 |Quantaware Distillation Training  |
