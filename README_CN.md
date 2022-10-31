[English](README.md) | 简体中文

![⚡️FastDeploy](https://user-images.githubusercontent.com/31974251/185771818-5d4423cd-c94c-4a49-9894-bc7a8d1c29d0.png)

</p>

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

**⚡️FastDeploy**是一款**易用高效**的推理部署开发套件。覆盖业界🔥**热门AI模型**并提供📦**开箱即用**的部署体验，包括图像分类、目标检测、图像分割、人脸检测、人脸识别、人体关键点识别、文字识别、语义理解等多任务，满足开发者**多场景**，**多硬件**、**多平台**的产业部署需求。

| Potrait Segmentation                                                                                        | Image Matting                                                                                             | Semantic Segmentation                                                                                                                                                                                                      | Real-Time Matting                                                                                                                                                                                                                         |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">                                                                                                                    | <img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px">                                                                                                                                 |
| **OCR**                 |  **Behavior Recognition**           | **Object Detection**                  |**Pose Estimation**
| <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"> |<img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px"  >                                                                                                                              |<img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px">  |
| **Face Alignment**                                                                                        | **3D Object Detection**                                                                                        |  **Face Editing**                                                                                                                                                                                                           | **Image Animation**  
| <img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188270227-1a4671b3-0123-46ab-8d0f-0e4132ae8ec0.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054663-b0c9c037-6d12-4e90-a7e4-e9abf4cf9b97.gif' height="126px" width="126px">  |  <img src='https://user-images.githubusercontent.com/54695910/188056800-2190e05e-ad1f-40ef-bf71-df24c3407b2d.gif' height="126px" width="190px">

## 近期更新

- 🔥 **2022.10.15：Release FastDeploy [release v0.3.0](https://github.com/PaddlePaddle/FastDeploy/tree/release%2F0.3.0)** <br>
  - **New server-side deployment upgrade:更快的推理性能，一键量化，更多的视觉和NLP模型**
       - 集成 OpenVINO 推理引擎，并且保证了使用 OpenVINO 与 使用 TensorRT、ONNX Runtime、 Paddle Inference一致的开发体验；
       - 提供[一键模型量化工具](tools/quantization)，支持YOLOv7、YOLOv6、YOLOv5等视觉模型，在CPU和GPU推理速度可提升1.5～2倍；
       - 新增加 PP-OCRv3, PP-OCRv2, PP-Matting, PP-HumanMatting, ModNet 等视觉模型并提供[端到端部署示例](examples/vision)；
       - 新增加NLP信息抽取模型 UIE 并提供[端到端部署示例](examples/text/uie).
       - 

- 🔥 **2022.8.18：发布FastDeploy [release v0.2.0](https://github.com/PaddlePaddle/FastDeploy/tree/release%2F0.2.0)** <br>
    - **服务端部署全新升级：更快的推理性能，更多的视觉模型支持**  
        - 发布基于x86 CPU、NVIDIA GPU的高性能推理引擎SDK，推理速度大幅提升
        - 集成Paddle Inference、ONNX Runtime、TensorRT等推理引擎并提供统一的部署体验
        - 支持YOLOv7、YOLOv6、YOLOv5、PP-YOLOE等全系列目标检测模型并提供[端到端部署示例](examples/vision/detection/)
        - 支持人脸检测、人脸识别、实时人像抠图、图像分割等40+重点模型及[Demo示例](examples/vision/)
        - 支持Python和C++两种语言部署
    - **边缘移动端部署新增瑞芯微、晶晨、恩智浦等NPU芯片部署能力**
        - 发布轻量化目标检测[Picodet-NPU部署Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)，提供极致的INT8全量化推理能力

## 目录
* **服务端部署**
    * [Python SDK快速开始](#fastdeploy-quick-start-python)  
    * [C++ SDK快速开始](#fastdeploy-quick-start-cpp)
    * [服务端模型支持列表](#fastdeploy-server-models)
* **端侧部署**
    * [Paddle Lite NPU部署](#fastdeploy-edge-sdk-npu)
    * [端侧模型支持列表](#fastdeploy-edge-sdk)
* [社区交流](#fastdeploy-community)
* [Acknowledge](#fastdeploy-acknowledge)  
* [License](#fastdeploy-license)

## 服务端部署

### Python SDK快速开始
<div id="fastdeploy-quick-start-python"></div>

#### 快速安装

##### 前置依赖
- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

##### 安装GPU版本

```bash
pip install numpy opencv-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```
##### [Conda安装(推荐)](docs/quick_start/Python_prebuilt_wheels.md)
```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```
##### 安装CPU版本

```bash
pip install numpy opencv-python fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### Python 推理示例

* 准备模型和图片

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* 测试推理结果
```python
# GPU/TensorRT部署参考 examples/vision/detection/paddledetection/python
import cv2
import fastdeploy.vision as vision

model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")
im = cv2.imread("000000014439.jpg")
result = model.predict(im.copy())
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```
<details>
<summary>C++ SDK快速开始</summary>
    
### C++ SDK快速开始
<div id="fastdeploy-quick-start-cpp"></div>

#### 安装

- 参考[C++预编译库下载](docs/quick_start/CPP_prebuilt_libraries.md)文档  

#### C++ 推理示例

* 准备模型和图片

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* 测试推理结果

```C++
// GPU/TensorRT部署参考 examples/vision/detection/paddledetection/cpp
#include "fastdeploy/vision.h"

int main(int argc, char* argv[]) {
  namespace vision = fastdeploy::vision;
  auto model = vision::detection::PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                          "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                          "ppyoloe_crn_l_300e_coco/infer_cfg.yml");
  auto im = cv::imread("000000014439.jpg");

  vision::DetectionResult res;
  model.Predict(&im, &res);

  auto vis_im = vision::Visualize::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
}
```
</details>

更多部署案例请参考[视觉模型部署示例](examples/vision) .

### 服务端和web模型支持列表 🔥🔥🔥

<div id="fastdeploy-server-models"></div>

符号说明: (1)  ✅: 已经支持; (2) ❔: 未来支持; (3) ❌: 暂不支持; (4) --: 暂不考虑;<br>
链接说明：「模型列」会跳转到模型推理Demo代码


| 任务场景                         | 模型                                                                                   | API                                                                                                                               | Linux                 | Linux                    | Win                      | Win                      | Mac                     | Mac                   | Linux                      | Linux                       | Linux                       |   [web_demo](examples/application/js/web_demo)         |       [mini_program](examples/application/js/mini_program)         |
|:-----------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------:|:---------------------:|:--------------------------:|:---------------------------:|:--------------------------:|:---------------------------:|:---------------------------:|
| ---                           | ---                                                                                     | ---                                                                                                                               | <font size=2> X86 CPU | <font size=2> NVIDIA GPU | <font size=2> Intel  CPU | <font size=2> NVIDIA GPU | <font size=2> Intel CPU | <font size=2> Arm CPU | <font size=2>  AArch64 CPU | <font size=2> NVIDIA Jetson | <font size=2> Graphcore IPU |[Paddle.js](examples/application/js)| [Paddle.js](examples/application/js)|
| <font size=2> Classification | <font size=2> [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |   ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |✅|✅|
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ❌ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |❔|❔|
| <font size=2> Classification | <font size=2> [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLOE](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ✅  | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ✅  | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ✅  | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ❌ | ❌ | ❌ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ❌ | ❌ | ❌ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❌ | ❌ | ❌ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox) | <font size=2> [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7) | <font size=2> [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [meituan/YOLOv6](./examples/vision/detection/yolov6) | <font size=2> [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [ultralytics/YOLOv5](./examples/vision/detection/yolov5) | <font size=2> [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [WongKinYiu/YOLOR](./examples/vision/detection/yolor) | <font size=2> [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4) | <font size=2> [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite) | <font size=2> [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Detection | <font size=2> [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus) | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|
| <font size=2> OCR | <font size=2> [PaddleOCR/PP-OCRv2](./examples/vision/ocr) | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/ocr/PP-OCRv3/cpp)|  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|
| <font size=2> OCR | <font size=2> [PaddleOCR/PP-OCRv3](./examples/vision/ocr) | <font size=2> [Python](./examples/vision/ocr/PP-OCRv3/python)/[C++](./examples/vision/ocr/PP-OCRv3/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |✅|✅|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |✅|✅|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Detection | <font size=2> [biubug6/RetinaFace](./examples/vision/facedet/retinaface) | <font size=2> [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Detection | <font size=2> [Linzaer/UltraFace](./examples/vision/facedet/ultraface) | [<font size=2> Python](./examples/vision/facedet/ultraface/python)/[C++](./examples/vision/facedet/ultraface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> FaceDetection | <font size=2> [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face) | <font size=2> [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Detection | <font size=2> [insightface/SCRFD](./examples/vision/facedet/scrfd) | <font size=2> [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Recognition | <font size=2> [insightface/ArcFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Recognition | <font size=2> [insightface/CosFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Recognition | <font size=2> [insightface/PartialFC](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Face Recognition | <font size=2> [insightface/VPL](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ | ❌ |❔|❔|
| <font size=2> Matting | <font size=2> [ZHKKKe/MODNet](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|
| <font size=2> Matting | <font size=2> [PaddleSeg/PP-Matting](./examples/vision/matting/ppmatting) | <font size=2> [Python](./examples/vision/matting/ppmatting/python)/[C++](./examples/vision/matting/ppmatting/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|
| <font size=2> Matting | <font size=2> [PaddleSeg/PP-HumanMatting](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/ppmatting/python)/[C++](./examples/vision/matting/ppmatting/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|
| <font size=2> Matting | <font size=2> [PaddleSeg/ModNet](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/ppmatting/python)/[C++](./examples/vision/matting/ppmatting/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|
| <font size=2> Information Extraction | <font size=2> [PaddleNLP/UIE](./examples/text/uie) | <font size=2> [Python](./examples/text/uie/python)/[C++](./examples/text/uie/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅| ❌ |❔|❔|


## 端侧部署

<div id="fastdeploy-edge-doc"></div>

### Paddle Lite NPU部署

<div id="fastdeploy-edge-sdk-npu"></div>

- [瑞芯微-NPU/晶晨-NPU/恩智浦-NPU](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)

### 端侧模型支持列表

<div id="fastdeploy-edge-sdk"></div>

|  任务场景 |  模型     |  大小(MB) | Linux  | Android  | iOS    |Linux  | Linux  | Linux    |更新中...|
|:------------------:|:----------------------------:|:---------------------:|:---------------------:|:----------------------:|:---------------------:| :------------------:|:----------------------------:|:---------------------:|:---------------------:|
| ---                | ---                          | ---                   | ARM CPU |  ARM CPU | ARM CPU |瑞芯微NPU<br>RV1109<br>RV1126<br>RK1808 | 晶晨NPU <br>A311D<br>S905D<br>C308X  | 恩智浦NPU<br>  i.MX 8M Plus    |更新中...｜
| Classification     | PP-LCNet                     | 11.9                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | PP-LCNetv2                   | 26.6                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | EfficientNet                 | 31.4                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | GhostNet                     | 20.8                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | MobileNetV1                  | 17                    | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | MobileNetV2                  | 14.2                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | MobileNetV3                  | 22                    | ✅                     | ✅                      | ✅                     |❔  | ❔  | ❔  |❔|
| Classification     | ShuffleNetV2                 | 9.2                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | SqueezeNetV1.1               | 5                     | ✅                     | ✅                      | ✅                     |
| Classification     | Inceptionv3                  | 95.5                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | PP-HGNet                     | 59                    | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | SwinTransformer_224_win7     | 352.7                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-PicoDet_s_320_coco        | 4.1                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-PicoDet_s_320_lcnet       | 4.9                   | ✅                     | ✅                      | ✅                     |✅   | ✅   | ✅     | ❔|
| Detection          | CenterNet                    | 4.8                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | YOLOv3_MobileNetV3           | 94.6                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLO_tiny_650e_coco       | 4.4                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | SSD_MobileNetV1_300_120e_voc | 23.3                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLO_ResNet50vd           | 188.5                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLOv2_ResNet50vd         | 218.7                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLO_crn_l_300e_coco      | 209.1                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | YOLOv5s                      | 29.3                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Face Detection      | BlazeFace                    | 1.5                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Face Detection      | RetinaFace                   | 1.7                   | ✅                     | ❌                      | ❌                     |--  | --  | --    |--|
| Keypoint Detection | PP-TinyPose                  | 5.5                   | ✅                     | ✅                      | ✅                     |❔ | ❔ | ❔ |❔|
| Segmentation       | PP-LiteSeg(STDC1)            | 32.2                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | PP-HumanSeg-Lite             | 0.556                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | HRNet-w18                    | 38.7                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | PP-HumanSeg-Server           | 107.2                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | Unet                         | 53.7                  | ❌                     | ✅                      | ❌                     |--  | --  | --    |--|
| OCR                | PP-OCRv1                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| OCR                | PP-OCRv2                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| OCR                | PP-OCRv3                     | 2.4+10.6              | ✅                     | ✅                      | ✅                     |❔ | ❔ | ❔  |❔|
| OCR                | PP-OCRv3-tiny                | 2.4+10.7              | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|

## 社区交流

<div id="fastdeploy-community"></div>

- **加入社区👬：** 微信扫描二维码，进入**FastDeploy技术交流群**

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/198922678-7f401024-9caf-4d18-bcdc-1298c9e80453.png"  width = "225" height = "225" />
</div>

## Acknowledge

<div id="fastdeploy-acknowledge"></div>

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，在此表示感谢。

## License

<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
