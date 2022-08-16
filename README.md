
![⚡️FastDeploy](docs/logo/fastdeploy-opaque.png)
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

**⚡️FastDeploy**是一款**简单易用**的推理部署工具箱。覆盖业界主流**优质预训练模型**并提供**开箱即用**的开发体验，包括图像分类、目标检测、图像分割、人脸检测、人体关键点识别、文字识别等多任务，满足开发者**多场景**，**多硬件**、**多平台**的快速部署需求。

## 发版历史
- [v0.2.0] 2022.08.18 全面开源服务端部署代码，支持40+视觉模型在CPU/GPU，以及通过TensorRT加速部署

## 内容目录
* [服务端模型列表](#fastdeploy-server-models)
* [服务端快速开始](#fastdeploy-quick-start)  
  * [快速安装](#fastdeploy-quick-start)
  * [Python预测示例](#fastdeploy-quick-start-python)  
  * [C++预测示例](#fastdeploy-quick-start-cpp)
* [轻量化SDK快速实现端侧AI推理部署](#fastdeploy-edge-sdk)
  * [边缘侧部署](#fastdeploy-edge-sdk-arm-linux)  
  * [移动端部署](#fastdeploy-edge-sdk-ios-android)  
  * [自定义模型部署](#fastdeploy-edge-sdk-custom)  
* [社区交流](#fastdeploy-community)
* [Acknowledge](#fastdeploy-acknowledge)  
* [License](#fastdeploy-license)
## 1. 服务端模型列表 🔥🔥🔥

<div id="fastdeploy-server-models"></div>

符号说明: (1)  ✅: 已经支持; (2) ❔: 计划未来支持; (3) ❌: 暂不支持; (4) contrib: 外部模型
| <font size=2> 任务场景 </font> | <font size=2> 模型  | <font size=2> API | <font size=2> Linux   |  <font size=2> Linux      |  <font size=2> Win   |  <font size=2> Win    |  <font size=2> Mac     | <font size=2>  Mac     | <font size=2> Linux |  
| :--------:  | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|  --- | --- |  --- |  <font size=2> X86 CPU |  <font size=2> NVIDIA GPU |  <font size=2> Intel  CPU |  <font size=2> NVIDIA GPU |  <font size=2> Intel CPU |  <font size=2> Arm CPU   | <font size=2> NVIDIA Jetson |
| <font size=2> Classification | <font size=2> [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/PPLCNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |   ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/PPLCNetv2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/PPHGNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Classification | <font size=2> [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PPYOLOE](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PPYOLO](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PPYOLOv2](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/YOLOX](./examples/vision/detection/yolox) | <font size=2> [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/YOLOv7](./examples/vision/detection/yolov7) | <font size=2> [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/YOLOv6](./examples/vision/detection/yolov6) | <font size=2> [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/YOLOv5](./examples/vision/detection/yolov5) | <font size=2> [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/YOLOR](./examples/vision/detection/yolor) | <font size=2> [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/ScaledYOLOv4](./examples/vision/detection/scaledyolov4) | <font size=2> [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/YOLOv5Lite](./examples/vision/detection/yolov5lite) | <font size=2> [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Detection | <font size=2> [Contrib/NanoDetPlus](./examples/vision/detection/nanodet_plus) | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PPLiteSeg](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PPHumanSegLite](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PPHumanSegServer](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceDetection | <font size=2> [Contrib/RetinaFace](./examples/vision/facedet/retinaface) | <font size=2> [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceDetection | <font size=2> [Contrib/UltraFace](./examples/vision/facedet/utltraface) | [<font size=2> Python](./examples/vision/facedet/utltraface/python)/[C++](./examples/vision/facedet/utltraface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceDetection | <font size=2> [Contrib/YOLOv5Face](./examples/vision/facedet/yolov5face) | <font size=2> [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceDetection | <font size=2> [Contrib/SCRFD](./examples/vision/facedet/scrfd) | <font size=2> [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [Contrib/ArcFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [Contrib/CosFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [Contrib/PartialFC](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> FaceRecognition | <font size=2> [Contrib/VPL](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |
| <font size=2> Matting | <font size=2> [Contrib/MODNet](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❔ | ❔ |


## 2. 服务端快速开始
<div id="fastdeploy-quick-start"></div>

<details>
<summary>💡 快速安装 FastDeploy Python/C++ 库 </summary>  

用户根据自己的python版本选择安装对应的wheel包，详细的wheel目录请参考 [python安装文档](docs/compile/prebuilt_wheels.md) .

```bash
pip install https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl
```
或获取C++预编译库，更多可用的预编译库请参考 [C++预编译库下载](docs/compile/prebuilt_libraries.md)
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/cpp/fastdeploy-linux-x64-0.2.0.tgz
```
准备目标检测模型和测试图片
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```
</details>

### 2.1 Python预测示例  
<div id="fastdeploy-quick-start-python"></div>

```python
import cv2
import fastdeploy.vision as vision

model = vision.detection.PPYOLOE("model.pdmodel", "model.pdiparams", "infer_cfg.yml")
im = cv2.imread("000000014439.jpg")
result = model.predict(im.copy())
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```
### 2.2 C++预测示例  
<div id="fastdeploy-quick-start-cpp"></div>

```C++
#include "fastdeploy/vision.h"

int main(int argc, char* argv[]) {
  namespace vision = fastdeploy::vision;
  auto model = vision::detection::PPYOLOE("model.pdmodel", "model.pdiparams", "infer_cfg.yml");
  auto im = cv::imread("000000014439.jpg");

  vision::DetectionResult res;
  model.Predict(&im, &res)

  auto vis_im = vision::Visualize::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
}
```
更多部署案例请参考[视觉模型部署示例](examples/vision) .

## 3. 轻量化SDK快速实现端侧AI推理部署 📱
<div id="fastdeploy-edge-sdk"></div>

| <font size=2> 任务场景 | <font size=2> 模型             | <font size=2>  大小(MB) | <font size=2>边缘端       | <font size=2>移动端       | <font size=2> 移动端     |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| ---               | ---                          | ---                   | <font size=2>  Linux  | <font size=2> Android  | <font size=2>  iOS    |
| ---              | ---                         | ---                   | <font size=2> ARM CPU | <font size=2>  ARM CPU | <font size=2> ARM CPU |
| Classification      | PP-LCNet                     | 11.9                  | ✅                     | ✅                      | ✅                     |
| Classification      | PP-LCNetv2                   | 26.6                  | ✅                     | ✅                      | ✅                     |
| Classification      | EfficientNet                 | 31.4                  | ✅                     | ✅                      | ✅                     |
| Classification      | GhostNet                     | 20.8                  | ✅                     | ✅                      | ✅                     |
| Classification      | MobileNetV1                  | 17                    | ✅                     | ✅                      | ✅                     |
| Classification      | MobileNetV2                  | 14.2                  | ✅                     | ✅                      | ✅                     |
| Classification      | MobileNetV3                  | 22                    | ✅                     | ✅                      | ✅                     |
| Classification      | ShuffleNetV2                 | 9.2                   | ✅                     | ✅                      | ✅                     |
| Classification      | SqueezeNetV1.1               | 5                     | ✅                     | ✅                      | ✅                     |
| Classification      | Inceptionv3                  | 95.5                  | ✅                     | ✅                      | ✅                     |
| Classification      | PP-HGNet                     | 59                    | ✅                     | ✅                      | ✅                     |
| Classification      | SwinTransformer_224_win7     | 352.7                 | ✅                     | ✅                      | ✅                     |
| Detection      | PP-PicoDet_s_320_coco        | 4.1                   | ✅                     | ✅                      | ✅                     |
| Detection      | PP-PicoDet_s_320_lcnet       | 4.9                   | ✅                     | ✅                      | ✅                     |
| Detection      | CenterNet                    | 4.8                   | ✅                     | ✅                      | ✅                     |
| Detection      | YOLOv3_MobileNetV3           | 94.6                  | ✅                     | ✅                      | ✅                     |
| Detection      | PP-YOLO_tiny_650e_coco       | 4.4                   | ✅                     | ✅                      | ✅                     |
| Detection      | SSD_MobileNetV1_300_120e_voc | 23.3                  | ✅                     | ✅                      | ✅                     |
| Detection      | PP-YOLO_ResNet50vd           | 188.5                 | ✅                     | ✅                      | ✅                     |
| Detection      | PP-YOLOv2_ResNet50vd         | 218.7                 | ✅                     | ✅                      | ✅                     |
| Detection      | PP-YOLO_crn_l_300e_coco      | 209.1                 | ✅                     | ✅                      | ✅                     |
| Detection      | YOLOv5s                      | 29.3                  | ✅                     | ✅                      | ✅                     |
| FaceDetection      | BlazeFace                    | 1.5                   | ✅                     | ✅                      | ✅                     |
| FaceDetection      | RetinaFace                   | 1.7                   | ✅                     | ❌                      | ❌                     |
| KeypointsDetection | PP-TinyPose                  | 5.5                   | ✅                     | ✅                      | ✅                     |
| Segmentation  | PP-LiteSeg(STDC1)            | 32.2                  | ✅                     | ✅                      | ✅                     |
| Segmentation  | PP-HumanSeg-Lite             | 0.556                 | ✅                     | ✅                      | ✅                     |
| Segmentation  | HRNet-w18                    | 38.7                  | ✅                     | ✅                      | ✅                     |
| Segmentation  | PP-HumanSeg-Server           | 107.2                 | ✅                     | ✅                      | ✅                     |
| Segmentation  | Unet                         | 53.7                  | ❌                     | ✅                      | ❌                     |
| OCR          | PP-OCRv1                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |
| OCR          | PP-OCRv2                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |
| OCR          | PP-OCRv3                     | 2.4+10.6              | ✅                     | ✅                      | ✅                     |
| OCR          | PP-OCRv3-tiny                | 2.4+10.7              | ✅                     | ✅                      | ✅                     |

### 3.1 边缘侧部署  
<div id="fastdeploy-edge-sdk-arm-linux"></div>

- ARM Linux 系统
  - [C++ Inference部署（含视频流）](./docs/ARM-Linux-CPP-SDK-Inference.md)
  - [C++ 服务化部署](./docs/ARM-Linux-CPP-SDK-Serving.md)
  - [Python Inference部署](./docs/ARM-Linux-Python-SDK-Inference.md)
  - [Python 服务化部署](./docs/ARM-Linux-Python-SDK-Serving.md)

### 3.2 移动端部署
<div id="fastdeploy-edge-sdk-ios-android"></div>

- [iOS 系统部署](./docs/iOS-SDK.md)
- [Android 系统部署](./docs/Android-SDK.md)  

### 3.3 自定义模型部署
<div id="fastdeploy-edge-sdk-custom"></div>

- [快速实现个性化模型替换](./docs/Replace-Model-With-Anther-One.md)

## 4. 社区交流
<div id="fastdeploy-community"></div>

- **加入社区👬：** 微信扫描二维码后，填写问卷加入交流群，与开发者共同讨论推理部署痛点问题

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854075-2c0f9997-ed18-4b17-9aaf-1b43266d3996.jpeg"  width = "200" height = "200" />
</div>

## 5. Acknowledge
<div id="fastdeploy-acknowledge"></div>

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，再次表示感谢。


## 6. License
<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
