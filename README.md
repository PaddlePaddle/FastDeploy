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

**⚡️FastDeploy**是一款**易用高效**的推理部署开发套件。覆盖业界主流**优质预训练模型**并提供**开箱即用**的部署体验，包括图像分类、目标检测、图像分割、人脸检测、人脸识别、人体关键点识别、文字识别等多任务，满足开发者**多场景**，**多硬件**、**多平台**的便捷高效的产业级部署需求。

## 近期更新

- 🔥 **2022.8.18：发布FastDeploy [release/v0.2.0](https://github.com/PaddlePaddle/FastDeploy/releases/tag/release%2F0.2.0)** <br>
    - **服务端部署全新升级：更快的推理性能，更多的视觉模型支持**  
        - 发布基于x86 CPU、NVIDIA GPU的高性能推理引擎SDK，推理速度大幅提升
        - 集成Paddle Inference、ONNX Runtime、TensorRT等推理引擎并提供统一的部署体验
        - 支持YOLOv7、YOLOv6、YOLOv5、PP-YOLOE等全系列目标检测模型并提供[端到端部署示例](examples/vision/detection/)
        - 支持人脸检测、人脸识别、实时人像抠图、图像分割等40+重点模型及[Demo示例](examples/vision/)
        - 支持Python和C++两种语言部署
    - **端侧部署新增瑞芯微、晶晨、恩智浦等NPU芯片部署能力**
        - 发布轻量化目标检测[Picodet-NPU部署Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)，提供低门槛INT8全量化能力

## 目录
* **服务端部署**
    * [FastDeploy Python SDK快速开始](#fastdeploy-quick-start-python)  
    * [FastDeploy C++ SDK快速开始](#fastdeploy-quick-start-cpp)
    * [服务端模型支持列表](#fastdeploy-server-models)
* **端侧部署**
    * [EasyEdge边缘端部署](#fastdeploy-edge-sdk-arm-linux)  
    * [EasyEdge移动端部署](#fastdeploy-edge-sdk-ios-android)  
    * [EasyEdge自定义模型部署](#fastdeploy-edge-sdk-custom)  
    * [Paddle Lite NPU部署](#fastdeploy-edge-sdk-npu)
    * [端侧模型支持列表](#fastdeploy-edge-sdk)
* [社区交流](#fastdeploy-community)
* [Acknowledge](#fastdeploy-acknowledge)  
* [License](#fastdeploy-license)

## 服务端部署

### FastDeploy Python SDK快速开始
<div id="fastdeploy-quick-start-python"></div>

#### 快速安装

##### 前置依赖
- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.8
- OS: Linux x86_64/macOS/Windows 10

##### 安装GPU版本

```bash
pip install numpy opencv-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
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

### FastDeploy C++ SDK快速开始
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
  model.Predict(&im, &res)

  auto vis_im = vision::Visualize::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
}
```

更多部署案例请参考[视觉模型部署示例](examples/vision) .

### 服务端模型支持列表 🔥🔥🔥

<div id="fastdeploy-server-models"></div>

符号说明: (1)  ✅: 已经支持; (2) ❔: 未来支持; (3) ❌: 暂不支持; (4) --: 暂不考虑;<br>
链接说明：「模型列」会跳转到模型推理Demo代码

| 任务场景 | 模型  | API | Linux   |   Linux      |   Win   |  Win    |   Mac     | Mac     |  Linux |   Linux |  
| :--------:  | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |:--------: |
|  --- | --- |  --- |  <font size=2> X86 CPU |  <font size=2> NVIDIA GPU |  <font size=2> Intel  CPU |  <font size=2> NVIDIA GPU |  <font size=2> Intel CPU |  <font size=2> Arm CPU   | <font size=2>  AArch64 CPU  | <font size=2> NVIDIA Jetson |
| <font size=2> Classification | <font size=2> [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |   ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLOE](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ❌ | ❌ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ❌ | ❌ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❌ | ❌ | ❔ |
| <font size=2> Detection | <font size=2> [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox) | <font size=2> [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7) | <font size=2> [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [meituan/YOLOv6](./examples/vision/detection/yolov6) | <font size=2> [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [ultralytics/YOLOv5](./examples/vision/detection/yolov5) | <font size=2> [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [WongKinYiu/YOLOR](./examples/vision/detection/yolor) | <font size=2> [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4) | <font size=2> [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite) | <font size=2> [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Detection | <font size=2> [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus) | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceDetection | <font size=2> [biubug6/RetinaFace](./examples/vision/facedet/retinaface) | <font size=2> [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceDetection | <font size=2> [Linzaer/UltraFace](./examples/vision/facedet/ultraface) | [<font size=2> Python](./examples/vision/facedet/ultraface/python)/[C++](./examples/vision/facedet/ultraface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceDetection | <font size=2> [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face) | <font size=2> [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceDetection | <font size=2> [deepinsight/SCRFD](./examples/vision/facedet/scrfd) | <font size=2> [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/ArcFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/CosFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/PartialFC](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/VPL](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |
| <font size=2> Matting | <font size=2> [ZHKKKe/MODNet](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ❔ |


## 端侧部署

<div id="fastdeploy-edge-doc"></div>

### EasyEdge边缘端部署

<div id="fastdeploy-edge-sdk-arm-linux"></div>

- ARM Linux 系统
  - [C++ Inference部署（含视频流）](./docs/ARM-CPU/ARM-Linux-CPP-SDK-Inference.md)
  - [C++ 服务化部署](./docs/ARM-CPU/ARM-Linux-CPP-SDK-Serving.md)
  - [Python Inference部署](./docs/ARM-CPU/ARM-Linux-Python-SDK-Inference.md)
  - [Python 服务化部署](./docs/ARM-CPU/ARM-Linux-Python-SDK-Serving.md)

### EasyEdge移动端部署

<div id="fastdeploy-edge-sdk-ios-android"></div>

- [iOS 系统部署](./docs/ARM-CPU/iOS-SDK.md)
- [Android 系统部署](./docs/ARM-CPU/Android-SDK.md)  

### EasyEdge自定义模型部署

<div id="fastdeploy-edge-sdk-custom"></div>

- [快速实现个性化模型替换](./docs/ARM-CPU/Replace-Model-With-Anther-One.md)

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
| FaceDetection      | BlazeFace                    | 1.5                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| FaceDetection      | RetinaFace                   | 1.7                   | ✅                     | ❌                      | ❌                     |--  | --  | --    |--|
| KeypointsDetection | PP-TinyPose                  | 5.5                   | ✅                     | ✅                      | ✅                     |❔ | ❔ | ❔ |❔|
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

- **加入社区👬：** 微信扫描二维码后，填写问卷加入交流群，与开发者共同讨论推理部署痛点问题

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854075-2c0f9997-ed18-4b17-9aaf-1b43266d3996.jpeg"  width = "200" height = "200" />
</div>

## Acknowledge

<div id="fastdeploy-acknowledge"></div>

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，在此表示感谢。

## License

<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
