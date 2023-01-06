English | [简体中文](README_CN.md)
# PaddleClas Quantification Model Deployment
FastDeploy supports the deployment of quantification models and provides a convenient tool for automatic model compression.
 Users can use it to deploy models after quantification or directly deploy quantized models provided by FastDeploy.

## FastDeploy one-click auto-compression tool
FastDeploy provides a one-click auto-compression tool that allows users to quantize models by simply entering a configuration file.
Refer to [one-click auto-compression tool](../../../../../tools/common_tools/auto_compression/) for details. 
Attention:The quantized classification model still requires the inference_cls.yaml file in the FP32 model folder. The model folder after personal quantification does not contain this yaml file. But users can copy this yaml file from the FP32 model folder to your quantized model folder.

## Download the quantized PaddleClas model
Users can also directly download the quantized models in the table below.

Benchmark table description:
- Runtime latency: model’s inference latency on multiple Runtimes, including CPU->GPU data copy, GPU inference, and GPU->CPU data copy time. It does not include the pre and post processing time of the model.
- End2End latency: model’s latency in the actual inference scenario, including the pre and post processing time of the model.
- Measured latency: The average latency after 1000 times of inference in milliseconds.
- INT8 + FP16: Enable FP16 inference for Runtime while inferring the INT8 quantification model
- INT8 + FP16 + PM: Use Pinned Memory to speed up the GPU->CPU data copy while inferring the INT8 quantization model with FP16 turned on.
- Maximum speedup ratio: Obtained by dividing the FP32 latency by the highest INT8 inference latency.
- The strategy is to use a few unlabeled data sets to train the model for quantification and to verify the accuracy on the full validation set. The INT8 accuracy does not represent the highest value.
- The CPU is Intel(R) Xeon(R) Gold 6271C, and the number of CPU threads is fixed to 1. The GPU is Tesla T4 with TensorRT version 8.4.15.

### Runtime Benchmark
| Model                 |Inference Backend            |Deployment Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  | Maximum Speedup Ratio    | FP32 Top1 | INT8 Top1 | Quantification Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | TensorRT         |    GPU    |  3.55 | 0.99|0.98|1.06  |      3.62      | 79.12  | 79.06 | Offline |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle-TensorRT  |    GPU    |  3.46 |None |0.87|1.03  |      3.98      | 79.12  | 79.06 | Offline |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | ONNX Runtime    |    CPU    |  76.14       |  35.43  |None|None  |     2.15        | 79.12  | 78.87| Offline|
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle Inference  |    CPU    |  76.21       |  24.01 |None|None  |     3.17       | 79.12  | 78.55 | Offline|
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | TensorRT  |    GPU    |     0.91 |   0.43 |0.49 | 0.54    |      2.12       |77.89 | 76.86 | Offline |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | Paddle-TensorRT   |    GPU    |  0.88|   None| 0.49|0.51 |      1.80      |77.89 | 76.86 | Offline |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | ONNX Runtime |    CPU    |     30.53   |   9.59|None|None    |     3.18       |77.89 | 75.09 |Offline |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        |  Paddle Inference  |    CPU    |     12.29  |   4.68  |     None|None|2.62       |77.89 | 71.36 |Offline |

### End2End Benchmark
| Model                 |Inference Backend          |Deployment Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency  | INT8+FP16+PM End2End Latency  | Maximum Speedup Ratio    | FP32 Top1 | INT8 Top1 | Quantification Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | TensorRT         |    GPU    |  4.92| 2.28|2.24|2.23 |      2.21     | 79.12  | 79.06 | Offline |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle-TensorRT  |    GPU    |  4.48|None |2.09|2.10 |      2.14   | 79.12  | 79.06 | Offline |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | ONNX Runtime    |    CPU    |  77.43    |  41.90 |None|None  |     1.85        | 79.12  | 78.87| Offline|
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | Paddle Inference  |    CPU    |   80.60     |  27.75 |None|None  |     2.90     | 79.12  | 78.55 | Offline|
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | TensorRT  |    GPU    |     2.19 |   1.48|1.57| 1.57   |      1.48     |77.89 | 76.86 | Offline |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | Paddle-TensorRT   |    GPU    |  2.04|   None| 1.47|1.45 |      1.41     |77.89 | 76.86 | Offline |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        | ONNX Runtime |    CPU    |     34.02  |   12.97|None|None    |    2.62       |77.89 | 75.09 |Offline |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)        |  Paddle Inference  |    CPU    |    16.31 |   7.42  |     None|None| 2.20      |77.89 | 71.36 |Offline |

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
