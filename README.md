# ⚡️FastDeploy

</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/FastDeploy?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://pypi.org/project/FastDeploy-python/"><img src="https://img.shields.io/pypi/dm/FastDeploy-python?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#服务器端> 服务器端 </a> |
  <a href=#端侧> 端侧 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

**⚡️FastDeploy**是一款**简单易用**的推理部署工具箱。覆盖业界主流**优质预训练模型**并提供**开箱即用**的开发体验，包括图像分类、目标检测、图像分割、人脸检测、人体关键点识别、文字识别等多任务，满足开发者**多场景**，**多硬件**、**多平台**的快速部署需求。

## News 📢

* 🔥 2022.8.15 [**⚡️FastDeploy v0.2.0**](https://github.com/PaddlePaddle/FastDeploy/releases/tag/release%2F0.2.0)测试版发布！🎉
  
  * 💎 升级服务器端（CPU/GPU/Jetson）SDK代码架构，速度SOTA
  * 😊 支持PyTorch模型部署，如YOLOv5、YOLOv6、YOLOv7等热门模型

## 特性

### 📦开箱即用的推理部署工具链，支持云边端、多硬件、多平台部署

- 支持 PIP 安装，一行命令快速下载SDK安装包，开箱即用
- 服务器与云端部署：
  - 跨平台：同时支持 Windows、Linux 操作系统
  - 多语言：提供 Python、C++ 多种语言部署示例
- 移动与边缘端侧部署：
  - 支持 iOS、Android 移动端部署
  - 支持 ARM Linux、NVIDIA Jetson 系列等边缘设备
- 覆盖主流AI硬件：
  - 支持 Intel CPU 系列（含酷睿、至强等）
  - 支持 ARM CPU 全系（含高通、MTK、RK等）
  - 支持 NVIDIA GPU 全系（含 A100、V100、T4、Jetson 等）



### 🤗丰富的预置模型与高性能部署示例

#### 服务器与云端（含Jetson）

| <font size=2> 任务场景              | <font size=2> 模型                                          | <font size=2>  大小(MB) | <font size=2>边缘端                        | <font size=2>服务器/云端             | <font size=2>服务器/云端  | <font size=2> 服务器/云端      | <font size=2> 服务器/云端  |
| ------------------------------- | --------------------------------------------------------- | --------------------- | --------------------------------------- | ------------------------------- | -------------------- | ------------------------- | --------------------- |
| ----                            | ----                                                      | ----                  | <font size=2> [Jetson](./doc/Jetson.md) | <font size=2> [X86 CPU](./doc/) | <font size=2>[GPU]() | <font size=2> [X86 CPU]() | <font size=2> [GPU]() |
| ----                            | ----                                                      | ----                  | Linux                                   | Windows                         | Linux                | Windows                   | Linux                 |
| Classfication                   |                                                           |                       |                                         |                                 |                      |                           |                       |
| Detection                       | [NanoDet-Plus](./model_zoo/vision/nanodet_plus/README.md) | 0.95~2.44             | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOR](./model_zoo/vison/yolor/README.md)                |                       | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOX](./model_zoo/vison/yolox/README.md)                |                       | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [Scaled-YOLOv4](./model_zoo/vison/scaledyolov4/README.md) | 4.9                   | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOv5](./model_zoo/vison/yolov5/README.md)              |                       | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOv5_Lite](./model_zoo/vison/yolov5lite/README.md)     | 94.6                  | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOv6](./model_zoo/vison/yolov6/README.md)              | 4.4                   | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOv7](./model_zoo/vison/yolov7/README.md)              | 23.3                  | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| <font size=2>Face Detection     | [UltraFace](./model_zoo/vison/ultraface/README.md)        | 1.04~1.1              | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [YOLOv5Face](./model_zoo/vison/yolov5face/README.md)      |                       | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| <font size=2>Face Localisation  | [RetinaFace](./model_zoo/vison/retinaface/README.md)      | 1.7M                  | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| <font size=2>Face Recognition   | [ArcFace](./model_zoo/vison/arcface/README.md)            | 1.7                   | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| <font size=2>Keypoint Detection | [SCRFD](./model_zoo/vison/scrfd/README.md)                | 5.5                   | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| <font size=2>Segmentation       | [PP-Seg](./model_zoo/vison/ppseg/README.md)               | 32.2                  | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| <font size=2>OCR                | [PP-OCRv1](./model_zoo/vison/ppocrv1/README.md)           | 2.3+4.4               | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [PP-OCRv2](./model_zoo/vison/ppocrv2/README.md)           | 2.3+4.4               | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
|                                 | [PP-OCRv3](./model_zoo/vison/ppocrv3/README.md)           | 2.4+10.6              | ✅                                       | ✅                               | ✅                    | ✅                         | ✅                     |
| </font>                         |                                                           |                       |                                         |                                 |                      |                           |                       |



#### 快速开始（服务器与云端部署）

开发者可以通过pip安装`fastdeploy-python`来获取最新的下载链接

- 环境依赖
  
  python >= 3.6

- 安装方式

```
pip install fastdeploy-python --upgrade
```

- 使用方式
  
  - 列出FastDeploy当前支持的所有模型
    
    ```
    fastdeploy --list_models
    ```
  
  - 下载模型在具体平台和对应硬件上的部署SDK以及示例
    
    ```
    fastdeploy --download_sdk \
             --model PP-PicoDet-s_320 \
             --platform Linux \
             --soc x86 \
             --save_dir .
    ```
  
  - 参数说明
    
    - `list_models`: 列出FastDeploy当前最新支持的所有模型
    - `download_sdk`: 下载模型在具体平台和对应硬件上的部署SDK以及示例
    - `model`: 模型名，如"PP-PicoDet-s_320"，可通过`list_models`查看所有的可选项
    - `platform`: 部署平台，支持 Windows/Linux/Android/iOS
    - `soc`: 部署硬件，支持 x86/x86-NVIDIA-GPU/ARM/Jetson
    - `save_dir`: SDK下载保存目录

### 📱轻量化SDK快速实现端侧AI推理部署


| <font size=2> 任务场景 | <font size=2> 模型             | <font size=2>  大小(MB) | <font size=2>边缘端       | <font size=2>移动端       | <font size=2> 移动端     |
| ------------------ | ---------------------------- | --------------------- | --------------------- | ---------------------- | --------------------- |
| ----               | ---                          | ---                   | <font size=2>  Linux  | <font size=2> Android  | <font size=2>  iOS    |
| -----              | ----                         | ---                   | <font size=2> ARM CPU | <font size=2>  ARM CPU | <font size=2> ARM CPU |
| Classfication      | PP-LCNet                     | 11.9                  | ✅                     | ✅                      | ✅                     |
|                    | PP-LCNetv2                   | 26.6                  | ✅                     | ✅                      | ✅                     |
|                    | EfficientNet                 | 31.4                  | ✅                     | ✅                      | ✅                     |
|                    | GhostNet                     | 20.8                  | ✅                     | ✅                      | ✅                     |
|                    | MobileNetV1                  | 17                    | ✅                     | ✅                      | ✅                     |
|                    | MobileNetV2                  | 14.2                  | ✅                     | ✅                      | ✅                     |
|                    | MobileNetV3                  | 22                    | ✅                     | ✅                      | ✅                     |
|                    | ShuffleNetV2                 | 9.2                   | ✅                     | ✅                      | ✅                     |
|                    | SqueezeNetV1.1               | 5                     | ✅                     | ✅                      | ✅                     |
|                    | Inceptionv3                  | 95.5                  | ✅                     | ✅                      | ✅                     |
|                    | PP-HGNet                     | 59                    | ✅                     | ✅                      | ✅                     |
|                    | SwinTransformer_224_win7     | 352.7                 | ✅                     | ✅                      | ✅                     |
| Detection          | PP-PicoDet_s_320_coco        | 4.1                   | ✅                     | ✅                      | ✅                     |
|                    | PP-PicoDet_s_320_lcnet       | 4.9                   | ✅                     | ✅                      | ✅                     |
|                    | CenterNet                    | 4.8                   | ✅                     | ✅                      | ✅                     |
|                    | YOLOv3_MobileNetV3           | 94.6                  | ✅                     | ✅                      | ✅                     |
|                    | PP-YOLO_tiny_650e_coco       | 4.4                   | ✅                     | ✅                      | ✅                     |
|                    | SSD_MobileNetV1_300_120e_voc | 23.3                  | ✅                     | ✅                      | ✅                     |
|                    | PP-YOLO_ResNet50vd           | 188.5                 | ✅                     | ✅                      | ✅                     |
|                    | PP-YOLOv2_ResNet50vd         | 218.7                 | ✅                     | ✅                      | ✅                     |
|                    | PP-YOLO_crn_l_300e_coco      | 209.1                 | ✅                     | ✅                      | ✅                     |
|                    | YOLOv5s                      | 29.3                  | ✅                     | ✅                      | ✅                     |
| Face Detection     | BlazeFace                    | 1.5                   | ✅                     | ✅                      | ✅                     |
| Face Localisation  | RetinaFace                   | 1.7                   | ✅                     | ❌                      | ❌                     |
| Keypoint Detection | PP-TinyPose                  | 5.5                   | ✅                     | ✅                      | ✅                     |
| Segmentation       | PP-LiteSeg(STDC1)            | 32.2                  | ✅                     | ✅                      | ✅                     |
|                    | PP-HumanSeg-Lite             | 0.556                 | ✅                     | ✅                      | ✅                     |
|                    | HRNet-w18                    | 38.7                  | ✅                     | ✅                      | ✅                     |
|                    | PP-HumanSeg-Server           | 107.2                 | ✅                     | ✅                      | ✅                     |
|                    | Unet                         | 53.7                  | ❌                     | ✅                      | ❌                     |
| OCR                | PP-OCRv1                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |
|                    | PP-OCRv2                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |
|                    | PP-OCRv3                     | 2.4+10.6              | ✅                     | ✅                      | ✅                     |
|                    | PP-OCRv3-tiny                | 2.4+10.7              | ✅                     | ✅                      | ✅                     |

 
#### SDK快速使用

##### 1.边缘侧部署

- ARM Linux 系统 
  - [C++ Inference部署（含视频流）](./docs/ARM-Linux-CPP-SDK-Inference.md)
  - [C++ 服务化部署](./docs/ARM-Linux-CPP-SDK-Serving.md)
  - [Python Inference部署](./docs/ARM-Linux-Python-SDK-Inference.md)
  - [Python 服务化部署](./docs/ARM-Linux-Python-SDK-Serving.md)

##### 2.移动端部署

- [iOS 系统部署](./docs/iOS-SDK.md)
- [Android 系统部署](./docs/Android-SDK.md)  

##### 3.自定义模型部署

- [快速实现个性化模型替换](./docs/Replace-Model-With-Anther-One.md)

## 社区交流

- **加入社区👬：** 微信扫描二维码后，填写问卷加入交流群，与开发者共同讨论推理部署痛点问题

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854075-2c0f9997-ed18-4b17-9aaf-1b43266d3996.jpeg"  width = "200" height = "200" />
</div>

## Acknowledge

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，再次表示感谢。

## License

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
