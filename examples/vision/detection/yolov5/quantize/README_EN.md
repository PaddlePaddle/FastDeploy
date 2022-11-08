# YOLOv5 Quantized Model Deployment

FastDeploy supports the deployment of quantized models and provides a one-click model quantization tool.
Users can use the one-click model quantization tool to quantize and deploy the models themselves or download the quantized models provided by FastDeploy directly for deployment.

## FastDeploy One-Click Model Quantization Tool

FastDeploy provides a one-click quantization tool that allows users to quantize a model simply with a configuration file.
For a detailed tutorial, please refer to: [One-Click Model Quantization Tool](... /... /... /... /... /... /... /tools/quantization/)

## Download Quantized YOLOv5s Model

Users can also directly download the quantized models in the table below for deployment.

| Model                                                                   | Inference Backend | Hardware | FP32 Inference Time Delay | INT8  Inference Time Delay | Acceleration ratio | FP32 mAP | INT8 mAP | Method                          |
| ----------------------------------------------------------------------- | ----------------- | -------- | ------------------------- | -------------------------- | ------------------ | -------- | -------- | ------------------------------- |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar) | TensorRT          | GPU      | 8.79                      | 5.17                       | 1.70               | 37.6     | 36.6     | Quantized distillation training |
| [YOLOv5s](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_quant.tar) | Paddle Inference  | CPU      | 217.05                    | 133.31                     | 1.63               | 37.6     | 36.8     | Quantized distillation training |

The data in the above table shows the end-to-end inference performance of FastDeploy deployment before and after model quantization.

- The test images are from COCO val2017.
- The inference time delay is the inference latency on different Runtime in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version 8.4.15, and the fixed CPU thread is 1 for all tests.

## More Detailed Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
