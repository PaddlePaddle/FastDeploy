简体中文 | [English](README.md)

# FastDeploy 服务化部署

## 简介

FastDeploy基于[Triton Inference Server](https://github.com/triton-inference-server/server)搭建了端到端的服务化部署。底层后端使用FastDeploy高性能Runtime模块，并串联FastDeploy前后处理模块实现端到端的服务化部署。具有快速部署、使用简单、性能卓越的特性。

> FastDeploy同时还提供了基于Python搭建的服务化部署能力，只需要通过Python即可启动服务，可参考[PaddleSeg部署示例](../examples/vision/segmentation/paddleseg/serving/simple_serving)了解其用法。
## 准备环境

### 环境要求
- Linux
- 如果使用GPU镜像， 要求NVIDIA Driver >= 470(如果是旧的Tesla架构GPU，如T4使用的NVIDIA Driver可以是418.40+、440.33+、450.51+、460.27+)

### 获取镜像

#### CPU镜像
CPU镜像仅支持Paddle/ONNX模型在CPU上进行服务化部署，支持的推理后端包括OpenVINO、Paddle Inference和ONNX Runtime
``` shell
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.4-cpu-only-21.10
```

#### GPU镜像
GPU镜像支持Paddle/ONNX模型在GPU/CPU上进行服务化部署，支持的推理后端包括OpenVINO、TensorRT、Paddle Inference和ONNX Runtime
```
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.4-gpu-cuda11.4-trt8.5-21.10
```

用户也可根据自身需求，参考如下文档自行编译镜像
- [FastDeploy服务化部署镜像编译说明](docs/zh_CN/compile.md)

## 其它文档
- [模型仓库目录说明](docs/zh_CN/model_repository.md) (说明如何准备模型仓库目录)
- [模型配置说明](docs/zh_CN/model_configuration.md)  (说明runtime的配置选项)
- [服务化部署示例](docs/zh_CN/demo.md) (服务化部署示例)
- [客户端访问说明](docs/zh_CN/client.md) (客户端访问说明)
- [Serving可视化部署](docs/zh_CN/vdl_management.md) (Serving可视化部署)


### 服务化部署示例

| 任务场景 | 模型  |
|---|---|
| Classification | [PaddleClas](../examples/vision/classification/paddleclas/serving/README.md) |
| Detection | [PaddleDetection](../examples/vision/detection/paddledetection/serving/README.md) |
| Detection | [ultralytics/YOLOv5](../examples/vision/detection/yolov5/serving/README.md) |
| NLP |	[PaddleNLP/ERNIE-3.0](../examples/text/ernie-3.0/serving/README.md)|
| NLP |	[PaddleNLP/UIE](../examples/text/uie/serving/README.md)|
| Speech |	[PaddleSpeech/PP-TTS](../examples/audio/pp-tts/serving/README.md)|
| OCR |	[PaddleOCR/PP-OCRv3](../examples/vision/ocr/PP-OCRv3/serving/README.md)|
