# YOLOv5量化模型部署
FastDeploy已支持部署量化模型,并提供一键模型量化的工具.
用户可以使用一键模型量化工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## FastDeploy一键模型量化工具
FastDeploy 提供了一键量化工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型量化工具](../../../../../tools/quantization/)

## 下载量化完成的YOLOv5s模型
用户也可以直接下载下表中的量化模型进行部署.(点击模型名字即可下载)

Benchmark表格说明:
- Rtuntime时延为模型在各种Runtime上的推理时延,包含CPU->GPU数据拷贝,GPU推理,GPU->CPU数据拷贝时间. 不包含模型各自的前后处理时间.
- 端到端时延为模型在实际推理场景中的时延, 包含模型的前后处理.
- 所测时延均为推理1000次后求得的平均值, 单位是毫秒.
- INT8 + FP16 为在推理INT8量化模型的同时, 给Runtime 开启FP16推理选项
- INT8 + FP16 + PM, 为在推理INT8量化模型和开启FP16的同时, 开启使用Pinned Memory的选项,可加速GPU->CPU数据拷贝的速度
- 最大加速比, 为FP32时延除以INT8推理的最快时延,得到最大加速比.
- 策略为量化蒸馏训练时, 采用少量无标签数据集训练得到量化模型, 并在全量验证集上验证精度, INT8精度并不代表最高的INT8精度.
- CPU为Intel(R) Xeon(R) Gold 6271C, 所有测试中固定CPU线程数为1.  GPU为Tesla T4, TensorRT版本8.4.15.


#### Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)               | TensorRT   |    GPU    |  7.87    | 4.51 |  4.31     | 3.17     |      2.48         | 37.6  | 36.7 | 量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)               | Paddle-TensorRT  |    GPU   |  7.99    |  None |  4.46    | 3.31     |      2.41         | 37.6  | 36.8 | 量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)                | ONNX Runtime   |    CPU    |  176.41      |    91.90   |  None |  None |      1.90        | 37.6  | 33.1 |量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)                | Paddle Inference|    CPU    |      213.73  |   130.19     |  None  | None |   1.64     |37.6 | 35.2 | 量化蒸馏训练 |

#### 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)               | TensorRT   |    GPU    |  24.61   | 21.20 |  20.78     | 20.94     |      1.18         | 37.6  | 36.7 | 量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)               | Paddle-TensorRT  |    GPU   |  23.53    |  None |  21.98    | 19.84     |      1.28        | 37.6  | 36.8 | 量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)                | ONNX Runtime   |    CPU    |  197.323      |    110.99   |  None |  None |      1.78        | 37.6  | 33.1 |量化蒸馏训练 |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar)                | Paddle Inference|    CPU    |      235.73  |   144.82     |  None  | None |   1.63     |37.6 | 35.2 | 量化蒸馏训练 |



## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
