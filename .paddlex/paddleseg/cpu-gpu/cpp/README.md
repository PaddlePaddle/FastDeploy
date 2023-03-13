# PaddleSeg C++部署示例

本目录下提供`infer.cc`快速完成PaddleSeg模型在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。

```bash
# 找到部署包内的模型路径，例如PP_LiteSeg

# 准备一张测试图片，例如test.jpg

# CPU推理
./infer_demo PP_LiteSeg test.jpg 0
# GPU推理
./infer_demo PP_LiteSeg test.jpg 1
# GPU上Paddle-TensorRT推理
./infer_demo PP_LiteSeg test.jpg 2
```

## 快速链接
- [PaddleSeg C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1segmentation.html)
