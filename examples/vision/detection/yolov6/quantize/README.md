[简体中文](README_CN.md) | English

# YOLOv6 Quantized Model Deployment

FastDeploy supports the deployment of quantized models and provides a one-click model quantization tool.
Users can use the one-click model quantization tool to quantize and deploy the models themselves or download the quantized models provided by FastDeploy directly for deployment.

## FastDeploy One-Click Model Quantization Tool

FastDeploy provides a one-click quantization tool that allows users to quantize a model simply with a configuration file.
For detailed tutorial, please refer to : [One-Click Model Quantization Tool](../../../../../tools/quantization/)

## Download Quantized YOLOv6s Model

Users can also directly download the quantized models in the table below for deployment.

| Model                                                                   | Inference Backend | Hardware | FP32  Inference Time Delay | INT8 Inference Time Delay | Acceleration ratio | FP32 mAP | INT8 mAP | Method                          |
| ----------------------------------------------------------------------- | ----------------- | -------- | -------------------------- | ------------------------- | ------------------ | -------- | -------- | ------------------------------- |
| [YOLOv6s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s_quant.tar) | TensorRT          | GPU      | 8.60                       | 5.16                      | 1.67               | 42.5     | 40.6     | Quantized distillation training |
| [YOLOv6s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s_quant.tar) | Paddle Inference  | CPU      | 356.62                     | 125.72                    | 2.84               | 42.5     | 41.2     | Quantized distillation training |

The data in the above table shows the end-to-end inference performance of FastDeploy deployment before and after model quantization.

- The test images are from COCO val2017.
- The inference time delay is the inference latency on different Runtimes in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version 8.4.15, and the fixed CPU thread is 1 for all tests.

## More Detailed Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
