简体中文 | [English](README_EN.md)

# FastDeploy 服务化部署

## 简介

FastDeploy基于[Triton Inference Server](https://github.com/triton-inference-server/server)搭建了端到端的服务化部署。底层后端使用FastDeploy高性能Runtime模块，并串联FastDeploy前后处理模块实现端到端的服务化部署。具有快速部署、使用简单、性能卓越的特性。

## 准备环境

### 环境要求
- Linux
- 如果使用GPU镜像， 要求NVIDIA Driver >= 470(如果是旧的Tesla架构GPU，如T4使用的NVIDIA Driver可以是418.40+、440.33+、450.51+、460.27+)

### 获取镜像

#### CPU镜像
CPU镜像仅支持Paddle/ONNX模型在CPU上进行服务化部署，支持的推理后端包括OpenVINO、Paddle Inference和ONNX Runtime
``` shell
docker pull paddlepaddle/fastdeploy:0.3.0-cpu-only-21.10
```

#### GPU镜像
GPU镜像支持Paddle/ONNX模型在GPU/CPU上进行服务化部署，支持的推理后端包括OpenVINO、TensorRT、Paddle Inference和ONNX Runtime
```
docker pull paddlepaddle/fastdeploy:0.3.0-gpu-cuda11.4-trt8.4-21.10
```

用户也可根据自身需求，参考如下文档自行编译镜像
- [FastDeploy服务化部署镜像编译说明](docs/zh_CN/compile.md)

## 其它文档
- [服务化模型目录说明](docs/zh_CN/model_repository.md) (说明如何准备模型目录)
- [服务化部署配置说明](docs/zh_CN/model_configuration.md)  (说明runtime的配置选项)
- [服务化部署示例](docs/zh_CN/demo.md)
  - [YOLOV5 检测任务](../examples/vision/detection/yolov5/serving/README.md)
