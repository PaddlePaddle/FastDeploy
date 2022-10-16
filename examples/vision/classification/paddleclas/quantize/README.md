# PaddleClas 量化模型部署
FastDeploy已支持部署量化模型,并提供一键模型量化的工具.
用户可以使用一键模型量化工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## FastDeploy一键模型量化工具
FastDeploy 提供了一键量化工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型量化工具](../../../../../tools/quantization/)
注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的inference_cls.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 下载量化完成的PaddleClas模型
用户也可以直接下载下表中的量化模型进行部署.
| 模型                 |推理后端            |部署硬件    | FP32推理时延    | INT8推理时延  | 加速比    | FP32 Top1 | INT8 Top1 |量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | ONNX Runtime         |    CPU    |   77.20       |  40.08     |     1.93        | 79.12  | 78.87|  离线量化|
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | TensorRT         |    GPU    |  3.70        | 1.80      |      2.06      | 79.12  | 79.06 | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)             | ONNX Runtime |    CPU    |     30.99   |   10.24    |     3.03        |77.89 | 75.09 |离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)             | TensorRT  |    GPU    |       1.80  |   0.58    |      3.10       |77.89 | 76.86 | 离线量化 |

上表中的数据, 为模型量化前后，在FastDeploy部署的端到端推理性能.
- 测试图片为ImageNet-2012验证集中的图片.
- 推理时延为端到端推理(包含前后处理)的平均时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1.

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
