# YOLOv5量化模型部署
FastDeploy已支持部署量化模型,并提供一键模型量化的工具.
用户可以使用一键模型量化工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## FastDeploy一键模型量化工具
FastDeploy 提供了一键量化工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型量化工具](../../../../../tools/quantization/)

## 下载量化完成的YOLOv5s模型
用户也可以直接下载下表中的量化模型进行部署.
| 模型                 |推理后端            |部署硬件    | FP32推理时延    | INT8推理时延  | 加速比    | FP32 mAP | INT8 mAP |量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)            | TensorRT         |    GPU    |  14.13        |  11.22      |      1.26         | 37.6  | 36.6 | 量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)             | Paddle Inference  |    CPU    |      226.36   |   152.27     |      1.48         |37.6 | 36.8 |量化蒸馏训练 |

上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试图片为COCO val2017中的图片.
- 推理时延为端到端推理(包含前后处理)的平均时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
