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

## 支持模型

| 任务场景 | 模型                                                         | X64 CPU | Nvidia-GPU | Nvidia-GPU TensorRT |
| -------- | ------------------------------------------------------------ | ------- | ---------- | ------------------- |
| 图像分类 | [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/PPLCNet](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | √       | √          | √                   |
| 目标检测 | [PaddleDetection/PPYOLOE](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/PPYOLO](./examples/vision/detection/paddledetection) | √       | √          | -                   |
|          | [PaddleDetection/PPYOLOv2](./examples/vision/detection/paddledetection) | √       | √          | -                   |
|          | [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | √       | √          | -                   |

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


#### 边缘侧部署

- ARM Linux 系统 
  - [C++ Inference部署（含视频流）](./docs/ARM-Linux-CPP-SDK-Inference.md)
  - [C++ 服务化部署](./docs/ARM-Linux-CPP-SDK-Serving.md)
  - [Python Inference部署](./docs/ARM-Linux-Python-SDK-Inference.md)
  - [Python 服务化部署](./docs/ARM-Linux-Python-SDK-Serving.md)

#### 移动端部署

- [iOS 系统部署](./docs/iOS-SDK.md)
- [Android 系统部署](./docs/Android-SDK.md)  

#### 自定义模型部署

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
