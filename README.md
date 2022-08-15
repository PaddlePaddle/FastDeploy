
![⚡️FastDeploy](docs/logo/fastdeploy-logo.png)
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

## 0. 发版历史
- [v0.2.0] 2022.08.18 全面开源服务端部署代码，支持40+视觉模型在CPU/GPU，以及通过GPU TensorRT加速部署

## 1. 内容目录
* [服务端模型列表](#fastdeploy-server-models)
* [服务端快速开始](#fastdeploy-quick-start)  
  * [Python预测示例](#fastdeploy-quick-start-python)  
  * [C++预测示例](#fastdeploy-quick-start-cpp)
* [更多服务端部署示例](#fastdeploy-server-cases)
* [轻量化SDK快速实现端侧AI推理部署](#fastdeploy-edge-sdk)
  * [边缘侧部署](#fastdeploy-edge-sdk-arm-linux)  
  * [移动端部署](#fastdeploy-edge-sdk-ios-android)  
  * [自定义模型部署](#fastdeploy-edge-sdk-custom)  
* [社区交流](#fastdeploy-community)
* [Acknowledge](#fastdeploy-acknowledge)  
* [License](#fastdeploy-license)
## 2. 服务端模型列表

<div id="fastdeploy-server-models"></div>

符号说明: (1) √: 已经支持, (2) ?: 待详细测试, (3) -: 暂不支持, (4) contrib: 非飞桨生态模型
| <font size=2> 任务场景 | <font size=2> 模型                                                         | <font size=2> API | <font size=2> CPU | <font size=2> GPU | <font size=2> Paddle | <font size=2> TRT | <font size=2> ORT |
| -------- | ------------------------------------------------------------ | ------- | ------- | ---------- | ---------| ---------| ---------|
| <font size=2> 图像分类 | <font size=2> [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/PPLCNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/PPLCNetv2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/PPHGNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 图像分类 | <font size=2> [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | √       | √          | √                   |   √        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/PPYOLOE](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                   | √        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/PPYOLO](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                    | -        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/PPYOLOv2](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                    | -        | ? |
| <font size=2> 目标检测 | <font size=2> [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | √       | √          | √                    | -        | ? |
| <font size=2> 目标检测 | <font size=2> [Contrib/YOLOX](./examples/vision/detection/yolox) | <font size=2> [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp) | √       | √          |  ?       | √          | √          |
| <font size=2> 目标检测 | <font size=2> [Contrib/YOLOv7](./examples/vision/detection/yolov7) | <font size=2> [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp) | √       | √          |  ?       | √      |√      |
| <font size=2> 目标检测 | <font size=2> [Contrib/YOLOv6](./examples/vision/detection/yolov6) | <font size=2> [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp) | √       | √          |  ?       | √      |√      |
| <font size=2> 目标检测 | <font size=2> [Contrib/YOLOv5](./examples/vision/detection/yolov5) | <font size=2> [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp) | √       | √          |  ?       | √      |√      |
| <font size=2> 目标检测 | <font size=2> [Contrib/YOLOR](./examples/vision/detection/yolor) | <font size=2> [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 目标检测 | <font size=2> [Contrib/ScaledYOLOv4](./examples/vision/detection/scaledyolov4) | <font size=2> [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp) | √       | √          | ?  | √      | √      |
| <font size=2> 目标检测 | <font size=2> [Contrib/YOLOv5Lite](./examples/vision/detection/yolov5lite) | <font size=2> [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp) | √       | √          |  ?     | √      | √      |
| <font size=2> 目标检测 | <font size=2> [Contrib/NanoDetPlus](./examples/vision/detection/nanodet_plus) | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp) | √       | √          |  ?       | √      |   √      |
| <font size=2> 图像分割 | <font size=2> [PaddleSeg/PPLiteSeg](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) | √       | √          | √       | √      |  ?      |
| <font size=2> 图像分割 | <font size=2> [PaddleSeg/PPHumanSegLite](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) | √       | √          | √ | √      |  ?      |
| <font size=2> 图像分割 | <font size=2> [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) | √       | √          | √                   |√      | ?       | ?       |
| <font size=2> 图像分割 | <font size=2> [PaddleSeg/PPHumanSegServer](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) | √       | √          | √                   |√      | ?       |
| <font size=2> 图像分割 | <font size=2> [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) | √       | √          | √                   |√      | ?       |
| <font size=2> 图像分割 | <font size=2> [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) | √       | √          | √                   |√      | ?      |
| <font size=2> 人脸检测 | <font size=2> [Contrib/RetinaFace](./examples/vision/facedet/retinaface) | <font size=2> [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp) | √       | √          | ?                   | √       | √       |
| <font size=2> 人脸检测 | <font size=2> [Contrib/UltraFace](./examples/vision/facedet/utltraface) | [<font size=2> Python](./examples/vision/facedet/utltraface/python)/[C++](./examples/vision/facedet/utltraface/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 人脸检测 | <font size=2> [Contrib/YOLOv5Face](./examples/vision/facedet/yolov5face) | <font size=2> [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 人脸检测 | <font size=2> [Contrib/SCRFD](./examples/vision/facedet/scrfd) | <font size=2> [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp) | √       | √          | ?                   | √        | √      |
| <font size=2> 人脸识别 | <font size=2> [Contrib/ArcFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 人脸识别 | <font size=2> [Contrib/CosFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 人脸识别 | <font size=2> [Contrib/PartialFC](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 人脸识别 | <font size=2> [Contrib/VPL](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) | √       | √          | ?                    |√      | √      |
| <font size=2> 人像抠图 | <font size=2> [Contrib/MODNet](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp) | √       | √          | ?                    | √       | √      |


## 3. 服务端快速开始
<div id="fastdeploy-quick-start"></div>

<details>
<summary>💡 安装FastDeploy Python/C++ </summary>  

用户根据开发环境选择安装版本，更多安装环境参考 [安装文档](docs/quick_start/install.md) .

```bash
pip install https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl
```
或获取C++预编译库，更多可用的预编译库请参考[C++预编译库下载](docs/compile/prebuilt_libraries.md)
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

### 3.1 Python预测示例  
<div id="fastdeploy-quick-start-python"></div>

```python
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
### 3.2 C++预测示例  
<div id="fastdeploy-quick-start-cpp"></div>

```C++
#include "fastdeploy/vision.h"

int main(int argc, char* argv[]) {
  namespce vision = fastdeploy::vision;
  auto model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel";
  auto params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams";
  auto config_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml";
  auto model = vision::detection::PPYOLOE(model_file, params_file, config_file);

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  vision::DetectionResult res;
  model.Predict(&im, &res)

  auto vis_im = vision::Visualize::VisDetection(im_bak, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_image.jpg" << std::endl;
}
```

## 4. 更多服务端部署示例  
<div id="fastdeploy-server-cases"></div>

FastDeploy提供了大量部署示例供开发者参考，支持模型在CPU、GPU以及TensorRT的部署

- [PaddleDetection模型部署](examples/vision/detection/paddledetection)
- [PaddleClas模型部署](examples/vision/classification/paddleclas)
- [PaddleSeg模型部署](examples/vision/segmentation/paddleseg)
- [YOLOv7部署](examples/vision/detection/yolov7)
- [YOLOv6部署](examples/vision/detection/yolov6)
- [YOLOv5部署](examples/vision/detection/yolov5)
- [人脸检测模型部署](examples/vision/facedet)
- [更多视觉模型部署示例...](examples/vision)

## 5. 📱轻量化SDK快速实现端侧AI推理部署
<div id="fastdeploy-edge-sdk"></div>


| <font size=2> 任务场景 | <font size=2> 模型             | <font size=2>  大小(MB) | <font size=2>边缘端       | <font size=2>移动端       | <font size=2> 移动端     |
| ------------------ | ---------------------------- | --------------------- | --------------------- | ---------------------- | --------------------- |
| ---               | ---                          | ---                   | <font size=2>  Linux  | <font size=2> Android  | <font size=2>  iOS    |
| ---              | ---                         | ---                   | <font size=2> ARM CPU | <font size=2>  ARM CPU | <font size=2> ARM CPU |
| 图像分类      | PP-LCNet                     | 11.9                  | ✅                     | ✅                      | ✅                     |
| 图像分类      | PP-LCNetv2                   | 26.6                  | ✅                     | ✅                      | ✅                     |
| 图像分类      | EfficientNet                 | 31.4                  | ✅                     | ✅                      | ✅                     |
| 图像分类      | GhostNet                     | 20.8                  | ✅                     | ✅                      | ✅                     |
| 图像分类      | MobileNetV1                  | 17                    | ✅                     | ✅                      | ✅                     |
| 图像分类      | MobileNetV2                  | 14.2                  | ✅                     | ✅                      | ✅                     |
| 图像分类      | MobileNetV3                  | 22                    | ✅                     | ✅                      | ✅                     |
| 图像分类      | ShuffleNetV2                 | 9.2                   | ✅                     | ✅                      | ✅                     |
| 图像分类      | SqueezeNetV1.1               | 5                     | ✅                     | ✅                      | ✅                     |
| 图像分类      | Inceptionv3                  | 95.5                  | ✅                     | ✅                      | ✅                     |
| 图像分类      | PP-HGNet                     | 59                    | ✅                     | ✅                      | ✅                     |
| 图像分类      | SwinTransformer_224_win7     | 352.7                 | ✅                     | ✅                      | ✅                     |
| 目标检测      | PP-PicoDet_s_320_coco        | 4.1                   | ✅                     | ✅                      | ✅                     |
| 目标检测      | PP-PicoDet_s_320_lcnet       | 4.9                   | ✅                     | ✅                      | ✅                     |
| 目标检测      | CenterNet                    | 4.8                   | ✅                     | ✅                      | ✅                     |
| 目标检测      | YOLOv3_MobileNetV3           | 94.6                  | ✅                     | ✅                      | ✅                     |
| 目标检测      | PP-YOLO_tiny_650e_coco       | 4.4                   | ✅                     | ✅                      | ✅                     |
| 目标检测      | SSD_MobileNetV1_300_120e_voc | 23.3                  | ✅                     | ✅                      | ✅                     |
| 目标检测      | PP-YOLO_ResNet50vd           | 188.5                 | ✅                     | ✅                      | ✅                     |
| 目标检测      | PP-YOLOv2_ResNet50vd         | 218.7                 | ✅                     | ✅                      | ✅                     |
| 目标检测      | PP-YOLO_crn_l_300e_coco      | 209.1                 | ✅                     | ✅                      | ✅                     |
| 目标检测      | YOLOv5s                      | 29.3                  | ✅                     | ✅                      | ✅                     |
| 人脸检测      | BlazeFace                    | 1.5                   | ✅                     | ✅                      | ✅                     |
| 人脸检测      | RetinaFace                   | 1.7                   | ✅                     | ❌                      | ❌                     |
| 人体关键点检测 | PP-TinyPose                  | 5.5                   | ✅                     | ✅                      | ✅                     |
| 图像分割  | PP-LiteSeg(STDC1)            | 32.2                  | ✅                     | ✅                      | ✅                     |
| 图像分割  | PP-HumanSeg-Lite             | 0.556                 | ✅                     | ✅                      | ✅                     |
| 图像分割  | HRNet-w18                    | 38.7                  | ✅                     | ✅                      | ✅                     |
| 图像分割  | PP-HumanSeg-Server           | 107.2                 | ✅                     | ✅                      | ✅                     |
| 图像分割  | Unet                         | 53.7                  | ❌                     | ✅                      | ❌                     |
| OCR          | PP-OCRv1                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |
| OCR          | PP-OCRv2                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |
| OCR          | PP-OCRv3                     | 2.4+10.6              | ✅                     | ✅                      | ✅                     |
| OCR          | PP-OCRv3-tiny                | 2.4+10.7              | ✅                     | ✅                      | ✅                     |

### 5.1 边缘侧部署  
<div id="fastdeploy-edge-sdk-arm-linux"></div>

- ARM Linux 系统
  - [C++ Inference部署（含视频流）](./docs/ARM-Linux-CPP-SDK-Inference.md)
  - [C++ 服务化部署](./docs/ARM-Linux-CPP-SDK-Serving.md)
  - [Python Inference部署](./docs/ARM-Linux-Python-SDK-Inference.md)
  - [Python 服务化部署](./docs/ARM-Linux-Python-SDK-Serving.md)

### 5.2 移动端部署
<div id="fastdeploy-edge-sdk-ios-android"></div>

- [iOS 系统部署](./docs/iOS-SDK.md)
- [Android 系统部署](./docs/Android-SDK.md)  

### 5.3 自定义模型部署
<div id="fastdeploy-edge-sdk-custom"></div>

- [快速实现个性化模型替换](./docs/Replace-Model-With-Anther-One.md)

## 6. 社区交流
<div id="fastdeploy-community"></div>

- **加入社区👬：** 微信扫描二维码后，填写问卷加入交流群，与开发者共同讨论推理部署痛点问题

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854075-2c0f9997-ed18-4b17-9aaf-1b43266d3996.jpeg"  width = "200" height = "200" />
</div>

## 7. Acknowledge
<div id="fastdeploy-acknowledge"></div>

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，再次表示感谢。


## 8. License
<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
