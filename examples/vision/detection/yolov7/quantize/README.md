# YOLOv7 Quantized Model Deployment

FastDeploy supports the deployment of quantized models and provides a one-click model quantization tool.
Users can use the one-click model quantization tool to quantize and deploy the models themselves or download the quantized models provided by FastDeploy directly for deployment.

## FastDeploy One-Click Model Quantization Tool

FastDeploy provides a one-click quantization tool that allows users to quantize a model simply with a configuration file.
For detailed tutorial, please refer to : [One-Click Model Quantization Tool](../../../../../tools/quantization/)

## Download Quantized YOLOv7 Model

Users can also directly download the quantized models in the table below for deployment.

| Model                                                                 | Inference Backend | Hardware | FP32 Inference Time Delay | FP32 Inference Time Delay | Acceleration ratio | FP32 mAP | INT8 mAP | Method                          |
| --------------------------------------------------------------------- | ----------------- | -------- | ------------------------- | ------------------------- | ------------------ | -------- | -------- | ------------------------------- |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_quant.tar) | TensorRT          | GPU      | 24.57                     | 9.40                      | 2.61               | 51.1     | 50.8     | Quantized distillation training |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_quant.tar) | Paddle Inference  | CPU      | 1022.55                   | 490.87                    | 2.08               | 51.1     | 46.3     | Quantized distillation training |

The data in the above table shows the end-to-end inference performance of FastDeploy deployment before and after model quantization.

- The test images are from COCO val2017.
- The inference time delay is the inference latency on different Runtimes in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version 8.4.15, and the fixed CPU thread is 1 for all tests.

## More Detailed Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
