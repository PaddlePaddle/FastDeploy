# FastDeploy

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
  <a href=#SDK安装> 安装 </a> |
  <a href=#SDK使用> 快速开始 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

**FastDeploy**是一款**简单易用**的推理部署工具箱。覆盖业界主流**优质预训练模型**并提供**开箱即用**的开发体验，包括图像分类、目标检测、图像分割、人脸检测、人体关键点识别、文字识别等多任务，满足开发者**多场景**，**多硬件**、**多平台**的快速部署需求。

## News 📢

* 🔥 2022.6.30 B站[飞桨直播课](https://space.bilibili.com/476867757)，FastDeploy天使用户邀测沟通会，与开发者共同讨论推理部署痛点问题。

* 🔥 2022.6.27 [**FastDeploy v0.1**](https://github.com/PaddlePaddle/FastDeploy/releases/tag/v0.1)邀测版发布！🎉
  * 💎 第一批发布对于40个重点模型在8种重点软硬件环境的支持的SDK
  * 😊 支持网页端、pip包两种下载使用方式


## 特性


### 📦开箱即用的推理部署工具链，支持云边端、多硬件、多平台部署
- 网页端点选下载、PIP 安装一行命令，快速下载多种类型SDK安装包
- 云端（含服务器、数据中心）：
    - 支持一行命令启动 Serving 服务（含网页图形化展示）
    - 支持一行命令启动图像、本地视频流、本地摄像头、网络视频流预测
    - 支持 Window、Linux 操作系统
    - 支持 Python、C++ 编程语言
- 边缘端：
    - 支持 NVIDIA Jetson 等边缘设备，支持视频流预测服务
- 端侧（含移动端）
    - 支持 iOS、Android 移动端
    - 支持 ARM CPU 端侧设备
- 支持主流硬件
    - 支持 Intel CPU 系列（含酷睿、至强等）
    - 支持 ARM CPU 全系（含高通、MTK、RK等）
    - 支持 NVIDIA GPU 全系（含 V100、T4、Jetson 等）

### 🤗丰富的预训练模型，轻松下载SDK搞定推理部署


<font size=0.5>

|<font size=2>   模型| <font size=2> 任务  |<font size=2>  大小(MB)  | <font size=2>端侧 | <font size=2>移动端 |<font size=2> 移动端 |<font size=2>边缘端 |<font size=2>服务器+云端 | <font size=2>服务器+云端 |<font size=2> 服务器+云端 |<font size=2> 服务器+云端 | 
|---|---|---|---|---|---|---|---|---|---|---|
|----- | ---- |----- |<font size=2>  Linux | <font size=2> Android |<font size=2>  iOS | <font size=2> Linux |<font size=2> Linux |<font size=2> Linux |<font size=2>  Windows  |<font size=2>  Windows  |
|----- | ---- |--- | <font size=2> ARM CPU |<font size=2>  ARM CPU | <font size=2> ARM CPU |<font size=2> Jetson |<font size=2> X86 CPU |<font size=2>  GPU  |<font size=2> X86 CPU |<font size=2>  GPU  |
| <font size=2> [PP-LCNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 11.9 |✅|✅|✅|✅|✅|✅|✅|✅|
| <font size=2> [PP-LCNetv2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 26.6 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [EfficientNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication |31.4 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [GhostNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 20.8 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [MobileNetV1](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 17 |✅|✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [MobileNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 14.2 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [MobileNetV3](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 22 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [ShuffleNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md)|Classfication | 9.2 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [SqueezeNetV1.1](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication |5 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [Inceptionv3](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication |95.5 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [PP-HGNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 59 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [ResNet50_vd](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 102.5 |❌|❌|❌|✅|✅|✅|✅|✅|
|<font size=2>  [SwinTransformer_224_win7](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/classification.md) |Classfication | 352.7 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [PP-PicoDet_s_320_coco](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 4.1 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [PP-PicoDet_s_320_lcnet](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 4.9 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [CenterNet](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection |4.8 |✅|✅|✅|✅ |✅ |✅|✅|✅|
|<font size=2>  [YOLOv3_MobileNetV3](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 94.6 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [PP-YOLO_tiny_650e_coco](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection |4.4 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [SSD_MobileNetV1_300_120e_voc](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 23.3 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [YOLOX_Nano_300e_coco](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 3.7 |❌|❌|❌|✅|✅ |✅|✅|✅|
|<font size=2> [PP-YOLO_ResNet50vd](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 188.5|✅ |✅ |✅ |✅ |✅ |✅|✅|✅|
|<font size=2>  [PP-YOLOv2_ResNet50vd](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 218.7 |✅|✅|✅|✅|✅ |✅|✅|✅|
|<font size=2>  [PP-YOLO_crn_l_300e_coco](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 209.1 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2> [YOLOv5s](https://github.com/ultralytics/yolov5) |Detection | 29.3|✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [Faster R-CNN_r50_fpn_1x_coco](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Detection | 167.2 |❌|❌|❌|✅|✅|✅|✅|✅|
|<font size=2> [BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Face Detection |1.5|✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2> [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) |Face Localisation |1.7| ✅|❌|❌|✅|✅|✅|✅|✅|
|<font size=2>  [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md) |Keypoint Detection| 5.5 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2> [PP-LiteSeg(STDC1)](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/pp_liteseg/README.md)|Segmentation | 32.2|✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [PP-HumanSeg-Lite](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/contrib/PP-HumanSeg/README_cn.md) |Segmentation | 0.556|✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [HRNet-w18](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/train/train_cn.md) |Segmentation | 38.7|✅|✅|✅|❌|✅|✅|✅|✅|
|<font size=2> [Mask R-CNN_r50_fpn_1x_coco](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/contrib/PP-HumanSeg/README_cn.md)|Segmentation| 107.2|❌|❌|❌|✅|✅|✅|✅|✅|
|<font size=2> [PP-HumanSeg-Server](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/contrib/PP-HumanSeg/README_cn.md)|Segmentation | 107.2|✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2> [Unet](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/train/train_cn.md) |Segmentation | 53.7|❌|✅|❌|❌|✅|✅|✅|❌|
|<font size=2> [Deeplabv3-ResNet50](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/train/train_cn.md)|Segmentation |156.5|❌|❌|❌|❌|✅|✅|✅|✅|
|<font size=2>  [PP-OCRv1](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/ppocr_introduction.md) |OCR | 2.3+4.4 |✅|✅|✅|✅|✅|✅|✅|✅|
|<font size=2>  [PP-OCRv2](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/ppocr_introduction.md) |OCR | 2.3+4.4 |✅|✅|✅|✅|✅|✅|✅|✅|
| <font size=2> [PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/PP-OCRv3_introduction.md) |OCR | 2.4+10.6 |✅|✅|✅|✅|✅|✅|✅|✅|
| <font size=2> [PP-OCRv3-tiny](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/models_list.md) |OCR |2.4+10.7 |✅|✅|✅|✅|✅|✅|✅|✅|
</font>


## SDK安装
    
### 方式1：网页版下载安装
    
- 可以登录[EasyEdge网页端](https://ai.baidu.com/easyedge/app/openSource)下载SDK 
    
### 方式2：pip安装

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
               --soc x86-NVIDIA-GPU \
               --save_dir .
    ```

    - 参数说明
        - `list_models`: 列出FastDeploy当前最新支持的所有模型
        - `download_sdk`: 下载模型在具体平台和对应硬件上的部署SDK以及示例
        - `model`: 模型名，如"PP-PicoDet-s_320"，可通过`list_models`查看所有的可选项
        - `platform`: 部署平台，支持 Windows/Linux/Android/iOS
        - `soc`: 部署硬件，支持Intel-x86_64/x86-NVIDIA-GPU/ARM/Jetson
        - `save_dir`: SDK下载保存目录

## SDK使用
### 1 云+服务器部署
   - Linux 系统(X86 CPU、NVIDIA GPU)
      - [C++ Inference部署（含视频流）](./docs/Linux-CPP-SDK-Inference.md)
      - [C++ 服务化部署](./docs/Linux-CPP-SDK-Serving.md)
      - [Python Inference部署](./docs/Linux-Python-SDK-Inference.md)
      - [Python 服务化部署](./docs/Linux-Python-SDK-Serving.md)
   - Window系统(X86 CPU、NVIDIA GPU)
      - [C++ Inference部署（含视频流）](./docs/Windows-CPP-SDK-Inference.md)
      - [C++ 服务化部署](./docs/Windows-CPP-SDK-Serving.md)
      - [Python Inference部署](./docs/Windows-Python-SDK-Inference.md)
      - [Python 服务化部署](./docs/Windows-Python-SDK-Serving.md)

### 2 边缘侧部署
   - ArmLinux 系统（NVIDIA Jetson Nano/TX2/Xavier）
      - [C++ Inference部署（含视频流）](./docs/Jetson-Linux-CPP-SDK-Inference.md)
      - [C++ 服务化部署](./docs/Jetson-Linux-CPP-SDK-Serving.md)

### 3 端侧部署
   - ArmLinux 系统(ARM CPU)    
      - [C++ Inference部署（含视频流）](./docs/ARM-Linux-CPP-SDK-Inference.md)
      - [C++ 服务化部署](./docs/ARM-Linux-CPP-SDK-Serving.md)
      - [Python Inference部署](./docs/ARM-Linux-Python-SDK-Inference.md)
      - [Python 服务化部署](./docs/ARM-Linux-Python-SDK-Serving.md)
 
### 4 移动端部署
   - [iOS 系统部署](./docs/iOS-SDK.md)
   - [Android 系统部署](./docs/Android-SDK.md)   
    
### 5 自定义模型部署
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
