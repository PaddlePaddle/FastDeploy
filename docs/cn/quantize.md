[English](../en/quantize.md) | 简体中文

# 量化加速
量化是一种流行的模型压缩方法，量化后的模型拥有更小的体积和更快的推理速度.
FastDeploy基于PaddleSlim的Auto Compression Toolkit(ACT), 给用户提供了一键模型自动化压缩的工具. FastDeploy一键模型自动压缩可包含多种策略, 目前主要采用离线量化和量化蒸馏训练.同时, FastDeploy支持部署压缩后的模型, 帮助用户实现推理加速. 本文主要描述量化模型在FastDeploy上的部署.


## FastDeploy 多个引擎和硬件支持量化模型部署
当前，FastDeploy中多个推理后端可以在不同硬件上支持量化模型的部署. 支持情况如下:

| 硬件/推理后端 | ONNX Runtime | Paddle Inference | TensorRT | Paddle-TensorRT |
| :-----------| :--------   | :--------------- | :------- | :------- |
|   CPU       |  支持        |  支持            |          |           |  
|   GPU       |             |                  | 支持      |    支持        |


## 模型量化

### 量化方法
基于PaddleSlim，目前FastDeploy一键模型自动压缩提供的的量化方法有量化蒸馏训练和离线量化，量化蒸馏训练通过模型训练来获得量化模型，离线量化不需要模型训练即可完成模型的量化。 FastDeploy 对两种方式产出的量化模型均能部署。

两种方法的主要对比如下表所示:
| 量化方法 | 量化过程耗时 | 量化模型精度 | 模型体积 | 推理速度 |
| :-----------| :--------| :-------| :------- | :------- |
|   离线量化      |  无需训练，耗时短 |  比量化蒸馏训练稍低       | 两者一致   | 两者一致   |  
|   量化蒸馏训练      |  需要训练，耗时稍高 |  较未量化模型有少量损失 | 两者一致   |两者一致   |  

### 使用FastDeploy一键模型自动化压缩工具来量化模型
FastDeploy基于PaddleSlim的Auto Compression Toolkit(ACT), 给用户提供了一键模型自动化压缩的工具，请参考如下文档进行一键模型自动化压缩。
- [FastDeploy 一键模型自动化压缩](../../tools/auto_compression/)
当用户获得产出的压缩模型之后，即可以使用FastDeploy来部署压缩模型。


## 量化模型 Benchmark

目前, FastDeploy支持自动化压缩,并完成部署测试的模型的Runtime Benchmark和端到端Benchmark如下所示.

Benchmark表格说明:
- Rtuntime时延为模型在各种Runtime上的推理时延,包含CPU->GPU数据拷贝,GPU推理,GPU->CPU数据拷贝时间. 不包含模型各自的前后处理时间.
- 端到端时延为模型在实际推理场景中的时延, 包含模型的前后处理.
- 所测时延均为推理1000次后求得的平均值, 单位是毫秒.
- INT8 + FP16 为在推理INT8量化模型的同时, 给Runtime 开启FP16推理选项
- INT8 + FP16 + PM, 为在推理INT8量化模型和开启FP16的同时, 开启使用Pinned Memory的选项,可加速GPU->CPU数据拷贝的速度
- 最大加速比, 为FP32时延除以INT8推理的最快时延,得到最大加速比.
- 策略为量化蒸馏训练时, 采用少量无标签数据集训练得到量化模型, 并在全量验证集上验证精度, INT8精度并不代表最高的INT8精度.
- CPU为Intel(R) Xeon(R) Gold 6271C, 所有测试中固定CPU线程数为1.  GPU为Tesla T4, TensorRT版本8.4.15.

### YOLO 系列
#### Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT   |    GPU    |  7.87    | 4.51 |  4.31     | 3.17     |      2.48         | 37.6  | 36.7 | 量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | Paddle-TensorRT  |    GPU   |  7.99    |  None |  4.46    | 3.31     |      2.41         | 37.6  | 36.8 | 量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime   |    CPU    |  176.41      |    91.90   |  None |  None |      1.90        | 37.6  | 33.1 |量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference|    CPU    |      213.73  |   130.19     |  None  | None |   1.64     |37.6 | 35.2 | 量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT  |    GPU    |       9.47    |   3.23    |  4.09      |2.81    |  3.37            | 42.5 | 40.7|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | Paddle-TensorRT |    GPU    |       9.31    | None|  4.17  | 2.95       |  3.16            | 42.5 | 40.7|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)          | ONNX Runtime     |    CPU    |   334.65     |  126.38      | None | None|     2.65   |42.5| 36.8|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |    352.87   |    123.12    |None | None|     2.87         |42.5| 40.8|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT   |    GPU    |     27.47    |  6.52   |  6.74| 5.19|    5.29       | 51.1| 50.4|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | Paddle-TensorRT |    GPU    |     27.87|None|6.91|5.86      |      4.76       | 51.1| 50.4|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     996.65        |  467.15 |None|None          |  2.13           | 51.1 | 43.3|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |     995.85  |     477.93|None|None      |   2.08         |51.1 | 46.2|量化蒸馏训练 |

#### 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT   |    GPU    |  24.61   | 21.20 |  20.78     | 20.94     |      1.18         | 37.6  | 36.7 | 量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | Paddle-TensorRT  |    GPU   |  23.53    |  None |  21.98    | 19.84     |      1.28        | 37.6  | 36.8 | 量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime   |    CPU    |  197.323      |    110.99   |  None |  None |      1.78        | 37.6  | 33.1 |量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference|    CPU    |      235.73  |   144.82     |  None  | None |   1.63     |37.6 | 35.2 | 量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT  |    GPU    |       15.66    |   11.30   |  10.25      |9.59   |  1.63           | 42.5 | 40.7|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | Paddle-TensorRT |    GPU    |       15.03   | None|  11.36 | 9.32       |  1.61            | 42.5 | 40.7|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)          | ONNX Runtime     |    CPU    |   348.21    |  126.38      | None | None| 2.82       |42.5| 36.8|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |    352.87   |    121.64    |None | None|    3.04       |42.5| 40.8|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT   |    GPU    |     36.47   |  18.81  |  20.33| 17.58|    2.07      | 51.1| 50.4|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | Paddle-TensorRT |    GPU    |     37.06|None|20.26|17.53    |      2.11      | 51.1| 50.4|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     988.85       |  478.08 |None|None          |  2.07          | 51.1 | 43.3|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |     1031.73 |     500.12|None|None      |   2.06         |51.1 | 46.2|量化蒸馏训练 |



### PaddleClas系列
#### Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 Top1 | INT8 Top1 | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  3.55 | 0.99|0.98|1.06  |      3.62      | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle-TensorRT  |    GPU    |  3.46 |None |0.87|1.03  |      3.98      | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime    |    CPU    |  76.14       |  35.43  |None|None  |     2.15        | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle Inference  |    CPU    |  76.21       |  24.01 |None|None  |     3.17       | 79.12  | 78.55 |  离线量化|
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | TensorRT  |    GPU    |     0.91 |   0.43 |0.49 | 0.54    |      2.12       |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | Paddle-TensorRT   |    GPU    |  0.88|   None| 0.49|0.51 |      1.80      |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | ONNX Runtime |    CPU    |     30.53   |   9.59|None|None    |     3.18       |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        |  Paddle Inference  |    CPU    |     12.29  |   4.68  |     None|None|2.62       |77.89 | 71.36 |离线量化 |

#### 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 Top1 | INT8 Top1 | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  4.92| 2.28|2.24|2.23 |      2.21     | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle-TensorRT  |    GPU    |  4.48|None |2.09|2.10 |      2.14   | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime    |    CPU    |  77.43    |  41.90 |None|None  |     1.85        | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | Paddle Inference  |    CPU    |   80.60     |  27.75 |None|None  |     2.90     | 79.12  | 78.55 |  离线量化|
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | TensorRT  |    GPU    |     2.19 |   1.48|1.57| 1.57   |      1.48     |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | Paddle-TensorRT   |    GPU    |  2.04|   None| 1.47|1.45 |      1.41     |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        | ONNX Runtime |    CPU    |     34.02  |   12.97|None|None    |    2.62       |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)        |  Paddle Inference  |    CPU    |    16.31 |   7.42  |     None|None| 2.20      |77.89 | 71.36 |离线量化 |



### PaddleDetection系列
#### Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | TensorRT         |    GPU    |  27.90 | 6.39 |6.44|5.95    |      4.67       | 51.4  | 50.7 | 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | Paddle-TensorRT |    GPU    |  30.89     |None  |  13.78 |14.01    |      2.24       | 51.4  | 50.5| 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | ONNX Runtime |    CPU    |     1057.82 |   449.52 |None|None    |      2.35        |51.4 | 50.0 |量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | Paddle Inference |    CPU    |     1235.54|   706.72 |None|None     |      1.75       |51.4 | 0.00 |量化蒸馏训练 |

NOTE:
- TensorRT比Paddle-TensorRT快的原因是在runtime移除了multiclass_nms3算子

#### 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | TensorRT         |    GPU    |  35.75 | 15.42 |20.70|20.85  |      2.32      | 51.4  | 50.7 | 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize )  | Paddle-TensorRT |    GPU    | 33.48    |None  |  18.47 |18.03   |     1.81       | 51.4  | 50.5| 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | ONNX Runtime |    CPU    |     1067.17 |   461.037 |None|None    |      2.31        |51.4 | 50.0 |量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](../../examples/vision/detection/paddledetection/quantize)  | Paddle Inference |    CPU    |    1246.15|   696.251 |None|None     |      1.79      |51.4 | 0.00 |量化蒸馏训练 |



### PaddleSeg系列
#### Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mIoU | INT8 mIoU | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](../../examples/vision/segmentation/paddleseg/quantize)  | Paddle Inference |    CPU    |     1138.04|   602.62 |None|None     |      1.89      |77.37 | 71.62 |量化蒸馏训练 |

#### 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mIoU | INT8 mIoU | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](../../examples/vision/segmentation/paddleseg/quantize)  | Paddle Inference |    CPU    |     4726.65|   4134.91|None|None     |      1.14      |77.37 | 71.62 |量化蒸馏训练 |
