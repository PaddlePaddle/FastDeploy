# PaddleClas CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

## 1. 运行部署示例

```bash
# 找到部署包内的模型路径，例如ResNet50

# 准备一张测试图片，例如test.jpg

# 在CPU上使用Paddle Inference推理
./infer_demo ResNet50 test.jpg 0
# 在CPU上使用OenVINO推理
./infer_demo ResNet50 test.jpg 1
# 在CPU上使用ONNX Runtime推理
./infer_demo ResNet50 test.jpg 2
# 在CPU上使用Paddle Lite推理
./infer_demo ResNet50 test.jpg 3
# 在GPU上使用Paddle Inference推理
./infer_demo ResNet50 test.jpg 4
# 在GPU上使用Paddle TensorRT推理
./infer_demo ResNet50 test.jpg 5
# 在GPU上使用ONNX Runtime推理
./infer_demo ResNet50 test.jpg 6
# 在GPU上使用Nvidia TensorRT推理
./infer_demo ResNet50 test.jpg 7
```

## 2. 部署示例选项说明  
在我们使用`infer_demo`时, 输入了3个参数, 分别为分类模型, 预测图片, 与最后一位的数字选项.
现在下表将解释最后一位数字选项的含义.
|数字选项|含义|
|:---:|:---:|
|0| 在CPU上使用Paddle Inference推理 |
|1| 在CPU上使用OpenVINO推理 |
|2| 在CPU上使用ONNX Runtime推理 |
|3| 在CPU上使用Paddle Lite推理 |
|4| 在GPU上使用Paddle Inference推理 |
|5| 在GPU上使用Paddle TensorRT推理 |
|6| 在GPU上使用ONNX Runtime推理 |
|7| 在GPU上使用Nvidia TensorRT推理 |

## 3. 更多指南
- [PaddleClas系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1classification.html)
