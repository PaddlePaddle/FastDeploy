# YOLOv7量化模型部署
FastDeploy 提供已经量化完成的yolov7模型，供用户体验推理加速.

## 下载量化完成的YOLOv7模型


| 模型                 |推理后端            |部署硬件    | FP32推理时延    | INT8推理时延  | 加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_quant.tar)            | TensorRT          |    GPU    |     30.43          |      15.40       |       1.98        | 51.1| 50.8| 量化蒸馏训练 |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_quant.tar)          | Paddle Inference  |    CPU    |          1015.70     |      562.41       |    1.82           |51.1 | 46.3| 量化蒸馏训练 |

上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试图片为COCO val2017中的图片.
- 推理时延为端到端推理(包含前后处理)的平均时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
