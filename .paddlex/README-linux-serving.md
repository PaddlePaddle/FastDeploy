# FastDeploy服务化部署SDK

## 一、SDK简介

### 1. SDK概览

FastDeploy是一款全场景、易用灵活、极致高效的AI推理部署工具，使用FastDeploy可以简单高效的在10+款硬件上对Paddle模型进行快速部署，本文档介绍在Linux下使用Python完成AI部署能力的介绍，更多能力PaddleX即将上线。

FastDeploy基于[Triton Inference Server](https://github.com/triton-inference-server/server)搭建了端到端的服务化部署。底层后端使用FastDeploy高性能Runtime模块，并串联FastDeploy前后处理模块实现端到端的服务化部署。具有快速部署、使用简单、性能卓越的特性。

> FastDeploy同时还提供了基于Python搭建的轻量服务化部署能力，只需要Python即可启动服务，可参考[PaddleSeg部署示例](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/segmentation/paddleseg/serving/simple_serving)了解其用法。


FastDeploy服务化部署SDK包由三部分组成：使用文档(README.md)、模型文件夹(model)、模型部署示例(example）。开发者可以快速验证模型的高性能部署，并将SDK集成到自己AI项目中。

```
Model_Name-FastDeploy-Linux-x86_64_CPU
├── README.md  # 使用文档，介绍SDK使用整体情况
├── model      # 训练好的模型文件
│   ├── inference.pdmodel         # 模型结构文件
│   ├── inference.pdiparams       # 模型参数文件
│   ├── inference_**.yaml         # 模型配置文件
│   ├── inference.pdiparams.info  
├── example                       # Serving部署示例
```

## 二、SDK快速使用

### 1. 环境准备

- Linux系统
- CPU镜像或GPU镜像
- 如果使用GPU镜像， 要求NVIDIA Driver >= 470(如果是旧的Tesla架构GPU，如T4使用的NVIDIA Driver可以是418.40+、440.33+、450.51+、460.27+)

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
- [FastDeploy服务化部署镜像编译说明](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/compile.md)

### 2. 运行部署示例

请参考example/目录下的README


## 三、FastDeploy部署其他参考文档

### 1. FastDeploy服务化部署其他文档

- [模型仓库目录说明](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_repository.md) (说明如何准备模型仓库目录)
- [模型配置说明](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_configuration.md)  (说明runtime的配置选项)
- [服务化部署示例](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/demo.md) (服务化部署示例)
- [客户端访问说明](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/client.md) (客户端访问说明)
- [Serving可视化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/vdl_management.md) (Serving可视化部署)

### 2. FastDeploy C++ API文档
https://baidu-paddle.github.io/fastdeploy-api/cpp/html/

### 3. FastDeploy Python API文档
https://baidu-paddle.github.io/fastdeploy-api/python/html/

### 4. 切换不同后端文档：
https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/how_to_change_backend.md

### 5. FastDeploy其他文档：
https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs

### 6. 库体积裁剪、性能极致优化等定制化需求
fastdeploy@baidu.com
