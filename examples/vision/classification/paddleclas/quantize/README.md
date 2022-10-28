[简体中文](README_CN.md) | English

# PaddleClas Quantized Model Deployment

FastDeploy supports the deployment of quantized models and provides a one-click model quantization tool.
Users can use the one-click model quantization tool to quantize and deploy the models themselves or download the quantized models provided by FastDeploy directly for deployment.

## FastDeploy One-Click Model Quantization Tool

FastDeploy provides a one-click quantization tool that allows users to quantize a model simply with a configuration file.
For a detailed tutorial, please refer to [One-Click Model Quantization Tool](../../../../../tools/quantization/)

Note: The quantized classification model still needs the inference_cls.yaml file in the FP32 model folder, while the model folder quantized by users does not contain this yaml file. Users can copy yaml file from the FP32 model folder to the quantized model folder.

## Download Quantized PaddleClas Model

Users can also directly download the quantized models in the table below for deployment.

| Model                                                                                   | Inference Backend | Hardware | FP32 Inference Time Delay | INT8  Inference Time Delay | Accleration ratio | FP32 Top1 | INT8 Top1 | Method                     |
| --------------------------------------------------------------------------------------- | ----------------- | -------- | ------------------------- | -------------------------- | ----------------- | --------- | --------- | -------------------------- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)           | ONNX Runtime      | CPU      | 77.20                     | 40.08                      | 1.93              | 79.12     | 78.87     | Post-training quantization |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)           | TensorRT          | GPU      | 3.70                      | 1.80                       | 2.06              | 79.12     | 79.06     | Post-training quantization |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar) | ONNX Runtime      | CPU      | 30.99                     | 10.24                      | 3.03              | 77.89     | 75.09     | Post-training quantization |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar) | TensorRT          | GPU      | 1.80                      | 0.58                       | 3.10              | 77.89     | 76.86     | Post-training quantization |

The data in the above table shows the end-to-end inference performance of FastDeploy deployment before and after model quantization.

- The test images are from the ImageNet-2012 validation set.
- The inference time delay is the average latency of end-to-end inference (including pre- and post-processing) in milliseconds.
- CPU is Intel(R) Xeon(R) Gold 6271C, GPU is Tesla T4, TensorRT version 8.4.15, and the number of fixed CPU threads is 1 in all tests.

## More Detailed Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
