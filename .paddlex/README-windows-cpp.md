# Windows C++ SDK

## 一、SDK简介

### 1. SDK概览

FastDeploy是一款全场景、易用灵活、极致高效的AI推理部署工具，使用FastDeploy可以简单高效的在10+款硬件上对PaddleSeg模型进行快速部署，本文档介绍在Linux下使用C++完成AI部署能力的介绍，更多能力PaddleX即将上线。

FastDeploy SDK 是基于 FastDeploy Runtime 多后端能力开发的，实现AI模型在各类硬件的高效推理部署。部署包由四部分组成：使用文档(README.md)、模型文件夹(model)、模型部署示例(example)、FastDeploy C++ SDK（CPU：fastdeploy-win-x64-0.0.0；GPU：fastdeploy-win-x64-gpu-0.0.0）。开发者可以快速验证模型的高性能部署，并将SDK集成到自己AI项目中。

```
Model_Name-FastDeploy-Linux-x86_64_CPU
├── README.md  # 使用文档，介绍SDK使用整体情况
├── model      # 训练好的模型文件
│   ├── inference.pdmodel         # 模型结构文件
│   ├── inference.pdiparams       # 模型参数文件
│   ├── inference_**.yaml         # 模型配置文件
│   ├── inference.pdiparams.info  
├── example    # C++部署示例
├── fastdeploy-win-x64-0.0.0  # FastDeploy C++ SDK
│   ├── libs         # FastDeploy库文件
│   ├── include      # FastDeploy头文件
│   ├── third_libs   # FastDeploy第三方依赖
│   ├── fastdeploy_init.bat
│   ├── FastDeploy.cmake
│   ├── FastDeployConfig.cmake
│   ├── FastDeployCSharp.cmake
│   ├── LICENSE
│   ├── openmp.cmake
│   ├── summary.cmake
│   ├── ThirdPartyNotices.txt
│   ├── utils.cmake
```

### 2. 适用范围

#### 2.1 主要功能

部署示例实现了模型高性能推理图像的功能，开发者可以基于[SDK API文档](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/)完成更多丰富的部署能力；视频推拉流即将支持；服务化部署，可以参考服务化部署文档。

#### 2.2 支持硬件

- CPU支持硬件类型：x86_64 CPU
- GPU支持硬件类型：全系列NVIDIA GPU

### 3. SDK版本说明

FastDeploy C++ SDK Version 0.0.0以FastDeploy源码develop分支编译，功能列表：
* 基础AI推理，支持C++、Python图像粒度推理。（即将支持C#、C）
* 支持用户基于FastDeploy API自定义开发。
* 支持用户自动编译不同后端，缩小库体积。
* 支持切换不同后端，放便快速对平台迁移。
* 集成高性能前后预处理库，保证AI推理端到端性能最优。


## 二、SDK快速使用

### 1. 环境准备

- cmake >= 3.12
- Visual Studio 16 2019
- GPU环境：CUDA 11.2, cuDNN 8.2

### 2. Linux编译部署示例

```
cd example
mkdir build
cd build
# 运行cmake命令需要指定FastDeploy SDK路径，FastDeploy SDK位于下载的部署包内
# 对于GPU环境，需要指定CUDA路径，例如-DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=..\..\fastdeploy-win-x64-0.0.0

msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64

```

在Windows上运行示例前，需要将FastDeploy依赖的库拷贝至可执行程序所在目录，或者配置环境变量。fastdeploy_init.bat脚本用于将所有依赖库拷贝至可执行程序所在目录：

```
cd fastdeploy-win-x64-0.0.0

fastdeploy_init.bat install %cd% ..\example\build\Release
```

### 3. 运行部署示例

请参考example/目录下的README

## 三、FastDeploy部署其他参考文档

### 1. C++ API文档
https://baidu-paddle.github.io/fastdeploy-api/cpp/html/

### 2. Python API文档
https://baidu-paddle.github.io/fastdeploy-api/python/html/

### 3. 切换不同后端文档：
https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/how_to_change_backend.md

### 4. FastDeploy其他文档：
https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs

### 5. 库体积裁剪、性能极致优化等定制化需求
fastdeploy@baidu.com
