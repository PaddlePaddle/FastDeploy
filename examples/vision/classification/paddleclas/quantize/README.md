# PaddleClas 量化模型部署
FastDeploy已支持部署量化模型,并提供一键模型量化的工具.
用户可以使用一键模型量化工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## FastDeploy一键模型量化工具
FastDeploy 提供了一键量化工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型量化工具](../../../../../tools/quantization/)
注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的inference_cls.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 下载量化完成的PaddleClas模型
用户也可以直接下载下表中的量化模型进行部署.

Benchmark表格说明:
- Rtuntime时延为模型在各种Runtime上的推理时延,包含CPU->GPU数据拷贝,GPU推理,GPU->CPU数据拷贝时间. 不包含模型各自的前后处理时间.
- 端到端时延为模型在实际推理场景中的时延, 包含模型的前后处理.
- 所测时延均为推理1000次后求得的平均值, 单位是毫秒.
- INT8 + FP16 为在推理INT8量化模型的同时, 给Runtime 开启FP16推理选项
- INT8 + FP16 + PM, 为在推理INT8量化模型和开启FP16的同时, 开启使用Pinned Memory的选项,可加速GPU->CPU数据拷贝的速度
- 最大加速比, 为FP32时延除以INT8推理的最快时延,得到最大加速比.
- 策略为量化蒸馏训练时, 采用少量无标签数据集训练得到量化模型, 并在全量验证集上验证精度, INT8精度并不代表最高的INT8精度.
- CPU为Intel(R) Xeon(R) Gold 6271C, 所有测试中固定CPU线程数为1.  GPU为Tesla T4, TensorRT版本8.4.15.

### Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 Top1 | INT8 Top1 | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | TensorRT         |    GPU    |  3.55 | 0.99|0.98|1.06  |      3.62      | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle-TensorRT  |    GPU    |  3.46 |None |0.87|1.03  |      3.98      | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | ONNX Runtime    |    CPU    |  76.14       |  35.43  |None|None  |     2.15        | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle Inference  |    CPU    |  76.21       |  24.01 |None|None  |     3.17       | 79.12  | 78.55 |  离线量化|
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | TensorRT  |    GPU    |     0.91 |   0.43 |0.49 | 0.54    |      2.12       |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | Paddle-TensorRT   |    GPU    |  0.88|   None| 0.49|0.51 |      1.80      |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | ONNX Runtime |    CPU    |     30.53   |   9.59|None|None    |     3.18       |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        |  Paddle Inference  |    CPU    |     12.29  |   4.68  |     None|None|2.62       |77.89 | 71.36 |离线量化 |

### 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 Top1 | INT8 Top1 | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | TensorRT         |    GPU    |  4.92| 2.28|2.24|2.23 |      2.21     | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle-TensorRT  |    GPU    |  4.48|None |2.09|2.10 |      2.14   | 79.12  | 79.06 | 离线量化 |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | ONNX Runtime    |    CPU    |  77.43    |  41.90 |None|None  |     1.85        | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle Inference  |    CPU    |   80.60     |  27.75 |None|None  |     2.90     | 79.12  | 78.55 |  离线量化|
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | TensorRT  |    GPU    |     2.19 |   1.48|1.57| 1.57   |      1.48     |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | Paddle-TensorRT   |    GPU    |  2.04|   None| 1.47|1.45 |      1.41     |77.89 | 76.86 | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | ONNX Runtime |    CPU    |     34.02  |   12.97|None|None    |    2.62       |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        |  Paddle Inference  |    CPU    |    16.31 |   7.42  |     None|None| 2.20      |77.89 | 71.36 |离线量化 |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
