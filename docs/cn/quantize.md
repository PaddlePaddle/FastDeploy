[English](../en/quantize.md) | 简体中文

# 量化加速
量化是一种流行的模型压缩方法，量化后的模型拥有更小的体积和更快的推理速度.
FastDeploy基于PaddleSlim, 集成了一键模型量化的工具, 同时, FastDeploy支持部署量化后的模型, 帮助用户实现推理加速.


## FastDeploy 多个引擎和硬件支持量化模型部署
当前，FastDeploy中多个推理后端可以在不同硬件上支持量化模型的部署. 支持情况如下:

| 硬件/推理后端 | ONNX Runtime | Paddle Inference | TensorRT |
| :-----------| :--------   | :--------------- | :------- |
|   CPU       |  支持        |  支持            |          |  
|   GPU       |             |                  | 支持      |


## 模型量化

### 量化方法
基于PaddleSlim, 目前FastDeploy提供的的量化方法有量化蒸馏训练和离线量化, 量化蒸馏训练通过模型训练来获得量化模型, 离线量化不需要模型训练即可完成模型的量化. FastDeploy 对两种方式产出的量化模型均能部署.

两种方法的主要对比如下表所示:
| 量化方法 | 量化过程耗时 | 量化模型精度 | 模型体积 | 推理速度 |
| :-----------| :--------| :-------| :------- | :------- |
|   离线量化      |  无需训练，耗时短 |  比量化蒸馏训练稍低       | 两者一致   | 两者一致   |  
|   量化蒸馏训练      |  需要训练，耗时稍高 |  较未量化模型有少量损失 | 两者一致   |两者一致   |  

### 用户使用FastDeploy一键模型量化工具来量化模型
Fastdeploy基于PaddleSlim, 为用户提供了一键模型量化的工具，请参考如下文档进行模型量化.
- [FastDeploy 一键模型量化](../../tools/quantization/)
当用户获得产出的量化模型之后，即可以使用FastDeploy来部署量化模型.


## 量化示例
目前, FastDeploy已支持的模型量化如下表所示:

### YOLO 系列
| 模型                 |推理后端            |部署硬件    | FP32推理时延/ms    | INT8推理时延/ms  | 加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)             | TensorRT         |    GPU    |  8.79       |  5.17     |      1.70         | 37.6  | 36.6 | 量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | ONNX Runtime     |    CPU    |  176.34      |    92.95   |      1.90        | 37.6  | 33.1 |量化蒸馏训练 |
| [YOLOv5s](../../examples/vision/detection/yolov5/quantize/)              | Paddle Inference  |    CPU    |      217.05  |   133.31     |     1.63         |37.6 | 36.8 | 量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | TensorRT         |    GPU    |       8.60       |   5.16         |  1.67            | 42.5 | 40.6|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)            | ONNX Runtime     |    CPU    |   338.60           |  128.58          |      2.60         |42.5| 36.1|量化蒸馏训练 |
| [YOLOv6s](../../examples/vision/detection/yolov6/quantize/)             | Paddle Inference  |    CPU    |        356.62     |    125.72        |     2.84         |42.5| 41.2|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)            | TensorRT          |    GPU    |     24.57         |      9.40     |      2.61       | 51.1| 50.8|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | ONNX Runtime     |    CPU    |     976.88         |  462.69          |  2.11            | 51.1 | 42.5|量化蒸馏训练 |
| [YOLOv7](../../examples/vision/detection/yolov7/quantize/)             | Paddle Inference  |    CPU    |         1022.55    |     490.87      |   2.08         |51.1 | 46.3|量化蒸馏训练 |


上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试数据为COCO2017验证集中的图片.
- 推理时延为在不同Runtime上推理的时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.


### PaddleClas系列
| 模型                 |推理后端            |部署硬件    | FP32推理时延/ms    | INT8推理时延/ms  | 加速比    | FP32 Top1 | INT8 Top1 |量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | ONNX Runtime         |    CPU    |  77.20       |  40.08     |     1.93        | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](../../examples/vision/classification/paddleclas/quantize/)            | TensorRT         |    GPU    |  3.70        | 1.80      |      2.06      | 79.12  | 79.06 | 离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)             | ONNX Runtime |    CPU    |     30.99   |   10.24    |     3.03        |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](../../examples/vision/classification/paddleclas/quantize/)             | TensorRT  |    GPU    |     1.80  |   0.58    |      3.10       |77.89 | 76.86 | 离线量化 |

上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试数据为ImageNet-2012验证集中的图片.
- 推理时延为在不同Runtime上推理的时延, 单位是毫秒（ms）.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.
