简体中文 | [English](README_EN.md)

# FastDeploy 服务化部署

## 简介

FastDeploy基于[Triton Inference Server](https://github.com/triton-inference-server/server)搭建了端到端的服务化部署。底层后端使用FastDeploy高性能Runtime模块，并串联FastDeploy前后处理模块实现端到端的服务化部署。具有快速部署、使用简单、性能卓越的特性。

## 准备环境

### 环境要求
- Linux
- CUDA >= 11.2

### 获取镜像

#### CPU镜像
CPU镜像仅支持Paddle/ONNX模型在CPU上进行服务化部署，支持的推理后端包括OpenVINO、Paddle Inference和ONNX Runtime
``` shell
docker pull xxxx:xxx
```

#### GPU镜像
GPU镜像支持Paddle/ONNX模型在GPU/CPU上进行服务化部署，支持的推理后端包括OpenVINO、TensorRT、Paddle Inference和ONNX Runtime
```
docker pull xxxx:xxxx
```

用户也可根据自身需求，参考如下文档自行编译镜像
- [FastDeploy服务化部署镜像编译说明]()

## 其它文档
- [服务化部署配置说明]()  (这里说明在服务化部署时，runtime的配置选项)
- [服务化部署示例]()  (这里提供包括FastDeploy的模型端到端服务化部署示例（可以是个表格）， 以及基于FastDeploy Runtime部署自己的Paddle/ONNX模型示例)
