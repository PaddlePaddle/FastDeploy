[English](../en/quantize.md) | 简体中文

# 量化加速
量化是一种流行的模型压缩方法，量化后的模型拥有更小的体积和更快的推理速度.
FastDeploy基于PaddleSlim, 集成了一键模型量化的工具, 同时, FastDeploy支持推理部署量化后的模型, 帮助用户实现推理加速.


## FastDeploy 多个引擎和硬件支持量化模型部署
当前，FastDeploy中多个推理后端可以在不同硬件上支持量化模型的部署. 支持情况如下:

| 硬件/推理后端 | ONNX Runtime | Paddle Inference | TensorRT |
| :-----------| :--------   | :--------------- | :------- |
|   CPU       |  支持        |  支持            |          |  
|   GPU       |             |                  | 支持      |


## 模型量化

### 量化方法
基于PaddleSlim，目前FastDeploy提供的的量化方法有量化蒸馏训练和离线量化，量化蒸馏训练通过模型训练来获得量化模型，离线量化不需要模型训练即可完成模型的量化。 FastDeploy 对两种方式产出的量化模型均能部署。

两种方法的主要对比如下表所示:
| 量化方法 | 量化过程耗时 | 量化模型精度 | 模型体积 | 推理速度 |
| :-----------| :--------| :-------| :------- | :------- |
|   离线量化      |  无需训练，耗时短 |  比量化蒸馏训练稍低       | 两者一致   | 两者一致   |  
|   量化蒸馏训练      |  需要训练，耗时稍高 |  较未量化模型有少量损失 | 两者一致   |两者一致   |  

### 使用FastDeploy一键模型量化工具来量化模型
Fastdeploy基于PaddleSlim, 为用户提供了一键模型量化的工具，请参考如下文档进行模型量化。
- [FastDeploy 一键模型量化](../../tools/quantization/)
当用户获得产出的量化模型之后，即可以使用FastDeploy来部署量化模型。


## 量化示例
目前, FastDeploy已支持的模型量化如下表所示:

### YOLO 系列
| 模型                 |推理后端            |部署硬件    | FP32推理时延    | INT8推理时延  | 加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT         |    GPU    |  14.13        |  11.22      |      1.26         | 37.6  | 36.6 | 量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime     |    CPU    |  183.68       |    100.39   |      1.83         | 37.6  | 33.1 |量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference  |    CPU    |      226.36   |   152.27     |      1.48         |37.6 | 36.8 | 量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT         |    GPU    |       12.89        |   8.92          |  1.45             | 42.5 | 40.6|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | ONNX Runtime     |    CPU    |   345.85            |  131.81           |      2.60         |42.5| 36.1|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |         366.41      |    131.70         |     2.78          |42.5| 41.2|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT          |    GPU    |     30.43          |      15.40       |       1.98        | 51.1| 50.8|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     971.27          |  471.88           |  2.06             | 51.1 | 42.5|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |          1015.70     |      562.41       |    1.82           |51.1 | 46.3|量化蒸馏训练 |

上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试数据为COCO2017验证集中的图片.
- 推理时延为端到端推理(包含前后处理)的平均时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.


### PaddleClas系列
| 模型                 |推理后端            |部署硬件    | FP32推理时延    | INT8推理时延  | 加速比    | FP32 Top1 | INT8 Top1 |量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime         |    CPU    |  86.87        |  59 .32     |      1.46         | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  7.85        |  5.42      |      1.45         | 79.12  | 79.06 | 离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)             | ONNX Runtime |    CPU    |      40.32   |   16.87     |      2.39         |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)             | TensorRT  |    GPU    |      5.10   |   3.35     |      1.52         |77.89 | 76.86 | 离线量化 |

上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试数据为ImageNet-2012验证集中的图片.
- 推理时延为端到端推理(包含前后处理)的平均时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.
