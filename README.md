English | [简体中文](README_ch.md)

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



**⚡️FastDeploy** is an **accessible and efficient** deployment Development Toolkit. It covers 🔥**hot AI models** in the industry and provides 📦**out-of-the-box** deployment experience. It covers image classification, object detection, image segmentation, face detection, face recognition, human keypoint detection, OCR, semantic understanding and other tasks to meet developers‘ industrial deployment needs for **multi-scenario**, **multi-hardware** and **multi-platform** .

| Potrait Segmentation                                                                                                                           | Image Matting                                                                                                                                  | Semantic Segmentation                                                                                                                            | Real-Time Matting                                                                                                                              |
|:----------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px"> |
| **OCR**                                                                                                                                        | **Behavior Recognition**                                                                                                                       | **Object Detection**                                                                                                                             | **Pose Estimation**                                                                                                                            |
| <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px"  > | <img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px"> |
| **Face Alignment**                                                                                                                             | **3D Object Detection**                                                                                                                        | **Face Editing**                                                                                                                                 | **Image Animation**                                                                                                                            |
| <img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188270227-1a4671b3-0123-46ab-8d0f-0e4132ae8ec0.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054663-b0c9c037-6d12-4e90-a7e4-e9abf4cf9b97.gif' height="126px" width="126px">   | <img src='https://user-images.githubusercontent.com/54695910/188056800-2190e05e-ad1f-40ef-bf71-df24c3407b2d.gif' height="126px" width="190px"> |

## Updates

- 🔥 **2022.8.18：Release FastDeploy [release/v0.2.0](https://github.com/PaddlePaddle/FastDeploy/releases/tag/release%2F0.2.0)** <br>
  - **New server-side deployment upgrade: faster inference performance, support more visual model**
    - Release high-performance inference engine SDK based on x86 CPUs and NVIDIA GPUs, with significant increase in inference speed
    - Integrate Paddle Inference, ONNXRuntime, TensorRT and other inference engines and provide a seamless deployment experience
    - Supports full range of object detection models such as YOLOv7, YOLOv6, YOLOv5, PP-YOLOE and provides [End-To-End Deployment Demos]](examples/vision/detection/)
    - Support over 40 key models and [Demo Examples](examples/vision/) including face detection, face recognition, real-time portrait matting, image segmentation.
    - Support deployment in both Python and C++
  - **Supports Rexchip, Amlogic, NXP and other NPU chip deployment capabilities on end-side deployment**
    - Release Lightweight Object Detection [Picodet-NPU Deployment Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_ detection), providing the full quantized inference capability for INT8.

## Contents

* **Server-side deployment**
  * [A Quick Start for Python SDK](#fastdeploy-quick-start-python)  
  * [A Quick Start for C++ SDK](#fastdeploy-quick-start-cpp)
  * [Supported Server-Side Model List](#fastdeploy-server-models)
* **End-side deployment**
  * [EasyEdge Edge-Side Deployment](#fastdeploy-edge-sdk-arm-linux)  
  * [EasyEdge Deployment on Mobile Devices](#fastdeploy-edge-sdk-ios-android)  
  * [EasyEdge Customised Model Deployment](#fastdeploy-edge-sdk-custom)  
  * [Paddle Lite NPU Deployment](#fastdeploy-edge-sdk-npu)
  * [Supported End-Side Model List](#fastdeploy-edge-sdk)
* [Community](#fastdeploy-community)
* [Acknowledge](#fastdeploy-acknowledge)  
* [License](#fastdeploy-license)

## Server-side deployment

### A Quick Start for Python SDK

<div id="fastdeploy-quick-start-python"></div>

#### Installation

##### Pre-dependency

- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.8
- OS: Linux x86_64/macOS/Windows 10

##### Install the GPU Version

```bash
pip install numpy opencv-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

##### [Conda Installation (Recommended)](docs/quick_start/Python_prebuilt_wheels.md)

```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

##### Install the CPU Version

```bash
pip install numpy opencv-python fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### Python Inference Example

* Prepare models and pictures

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference resultsTest inference results
  
  ```python
  # For deployment of GPU/TensorRT, please refer to examples/vision/detection/paddledetection/python
  import cv2
  import fastdeploy.vision as vision
  ```

model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")
im = cv2.imread("000000014439.jpg")
result = model.predict(im.copy())
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)

### A Quick Start for C++ SDK

#### Installation

- Please refer to [C++ Prebuilt Libraries Download](docs/quick_start/CPP_prebuilt_libraries.md)

#### C++ Inference Example

* Prepare models and pictures

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference results

```C++
// For GPU/TensorRT deployment, please refer to examples/vision/detection/paddledetection/cpp
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
```

### For more deployment models, please refer to [Visual Model Deployment Examples](examples/vision) .

 

### Supported Server-Side Model List🔥🔥🔥

<div id="fastdeploy-server-models"></div>

Notes:

 (1) ✅: already supported; (2) ❔: to be supported in the future; (3) ❌: not supported at the moment; (4) --: not considered at the moment;<br>Hyperlinks：Click model's name, the website will jump to the model inference demo code



| Task                          | Model                                                                                   | API                                                                                                                               | Linux                 | Linux                    | Win                      | Win                      | Mac                     | Mac                   | Linux                      | Linux                       |
|:-----------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------:|:---------------------:|:--------------------------:|:---------------------------:|
| ---                           | ---                                                                                     | ---                                                                                                                               | <font size=2> X86 CPU | <font size=2> NVIDIA GPU | <font size=2> Intel  CPU | <font size=2> NVIDIA GPU | <font size=2> Intel CPU | <font size=2> Arm CPU | <font size=2>  AArch64 CPU | <font size=2> NVIDIA Jetson |
| <font size=2> Classification  | <font size=2> [PaddleClas/ResNet50](./examples/vision/classification/paddleclas)        | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas)        | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas)      | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas)    | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/GhostNet](./examples/vision/classification/paddleclas)        | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas)     | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas)     | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas)     | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas)    | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas)     | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas)        | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Classification  | <font size=2> [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/PP-YOLOE](./examples/vision/detection/paddledetection)   | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection)    | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection)      | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection)     | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection)    | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ❌                     | ❌                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection)  | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ❌                     | ❌                          | ❔                           |
| <font size=2> Detection       | <font size=2> [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ❌                     | ❌                          | ❔                           |
| <font size=2> Detection       | <font size=2> [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox)           | <font size=2> [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp)                     | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7)                   | <font size=2> [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp)                   | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [meituan/YOLOv6](./examples/vision/detection/yolov6)                      | <font size=2> [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp)                   | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [ultralytics/YOLOv5](./examples/vision/detection/yolov5)                  | <font size=2> [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp)                   | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [WongKinYiu/YOLOR](./examples/vision/detection/yolor)                     | <font size=2> [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp)                     | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4)       | <font size=2> [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite)                | <font size=2> [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp)           | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Detection       | <font size=2> [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus)          | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Segmentation    | <font size=2> [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg)          | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Segmentation    | <font size=2> [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg)     | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Segmentation    | <font size=2> [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg)               | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Segmentation    | <font size=2> [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg)   | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Segmentation    | <font size=2> [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg)                | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Segmentation    | <font size=2> [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg)           | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceDetection   | <font size=2> [biubug6/RetinaFace](./examples/vision/facedet/retinaface)                | <font size=2> [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp)               | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceDetection   | <font size=2> [Linzaer/UltraFace](./examples/vision/facedet/ultraface)                  | [<font size=2> Python](./examples/vision/facedet/ultraface/python)/[C++](./examples/vision/facedet/ultraface/cpp)                 | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceDetection   | <font size=2> [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face)             | <font size=2> [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp)               | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceDetection   | <font size=2> [deepinsight/SCRFD](./examples/vision/facedet/scrfd)                      | <font size=2> [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp)                         | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/ArcFace](./examples/vision/faceid/insightface)               | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)               | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/CosFace](./examples/vision/faceid/insightface)               | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)               | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/PartialFC](./examples/vision/faceid/insightface)             | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)               | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> FaceRecognition | <font size=2> [deepinsight/VPL](./examples/vision/faceid/insightface)                   | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)               | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |
| <font size=2> Matting         | <font size=2> [ZHKKKe/MODNet](./examples/vision/matting/modnet)                         | <font size=2> [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp)                       | ✅                     | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           |

## Edge-Side Deployment

<div id="fastdeploy-edge-doc"></div>

### EasyEdge Edge-Side Deployment

<div id="fastdeploy-edge-sdk-arm-linux"></div>

- ARM Linux System
  - [C++ Inference Deployment（includes videostream）](./docs/arm_cpu/arm_linux_cpp_sdk_inference.md)
  - [C++ Serving Deployment](./docs/arm_cpu/arm_linux_cpp_sdk_serving.md)
  - [Python Inference Deployment](./docs/arm_cpu/arm_linux_python_sdk_inference.md)
  - [Python Serving Deploymen](./docs/arm_cpu/arm_linux_python_sdk_serving.md)

### EasyEdge Deployment on Mobile Devices

<div id="fastdeploy-edge-sdk-ios-android"></div>

- [iOS System Deployment](./docs/arm_cpu/ios_sdk.md)
- [Android System Deployment](./docs/arm_cpu/android_sdk.md)  

### EasyEdge Customized Deployment

<div id="fastdeploy-edge-sdk-custom"></div>

- [Replace Model With Another One](./docs/arm_cpu/replace_model_with_another_one.md)

### Paddle Lite NPU Deployment

<div id="fastdeploy-edge-sdk-npu"></div>

- [Rexchip-NPU / Amlogic-NPU / NXP-NPU](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)

### Supported Edge-Side Model List

<div id="fastdeploy-edge-sdk"></div>

|                    | Model                        | Size (MB) | Linux   | Android | iOS     | Linux                                     | Linux                                   | Linux                    | TBD...  |
|:------------------:|:----------------------------:|:---------:|:-------:|:-------:|:-------:|:-----------------------------------------:|:---------------------------------------:|:------------------------:|:-------:|
| ---                | ---                          | ---       | ARM CPU | ARM CPU | ARM CPU | Rexchip-NPU<br>RV1109<br>RV1126<br>RK1808 | Amlogic-NPU <br>A311D<br>S905D<br>C308X | NXPNPU<br>  i.MX 8M Plus | TBD...｜ |
| Classification     | PP-LCNet                     | 11.9      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | PP-LCNetv2                   | 26.6      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | EfficientNet                 | 31.4      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | GhostNet                     | 20.8      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | MobileNetV1                  | 17        | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | MobileNetV2                  | 14.2      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | MobileNetV3                  | 22        | ✅       | ✅       | ✅       | ❔                                         | ❔                                       | ❔                        | ❔       |
| Classification     | ShuffleNetV2                 | 9.2       | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | SqueezeNetV1.1               | 5         | ✅       | ✅       | ✅       |                                           |                                         |                          |         |
| Classification     | Inceptionv3                  | 95.5      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | PP-HGNet                     | 59        | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Classification     | SwinTransformer_224_win7     | 352.7     | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | PP-PicoDet_s_320_coco        | 4.1       | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | PP-PicoDet_s_320_lcnet       | 4.9       | ✅       | ✅       | ✅       | ✅                                         | ✅                                       | ✅                        | ❔       |
| Detection          | CenterNet                    | 4.8       | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | YOLOv3_MobileNetV3           | 94.6      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | PP-YOLO_tiny_650e_coco       | 4.4       | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | SSD_MobileNetV1_300_120e_voc | 23.3      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | PP-YOLO_ResNet50vd           | 188.5     | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | PP-YOLOv2_ResNet50vd         | 218.7     | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | PP-YOLO_crn_l_300e_coco      | 209.1     | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Detection          | YOLOv5s                      | 29.3      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| FaceDetection      | BlazeFace                    | 1.5       | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| FaceDetection      | RetinaFace                   | 1.7       | ✅       | ❌       | ❌       | --                                        | --                                      | --                       | --      |
| KeypointsDetection | PP-TinyPose                  | 5.5       | ✅       | ✅       | ✅       | ❔                                         | ❔                                       | ❔                        | ❔       |
| Segmentation       | PP-LiteSeg(STDC1)            | 32.2      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Segmentation       | PP-HumanSeg-Lite             | 0.556     | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Segmentation       | HRNet-w18                    | 38.7      | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Segmentation       | PP-HumanSeg-Server           | 107.2     | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| Segmentation       | Unet                         | 53.7      | ❌       | ✅       | ❌       | --                                        | --                                      | --                       | --      |
| OCR                | PP-OCRv1                     | 2.3+4.4   | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| OCR                | PP-OCRv2                     | 2.3+4.4   | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |
| OCR                | PP-OCRv3                     | 2.4+10.6  | ✅       | ✅       | ✅       | ❔                                         | ❔                                       | ❔                        | ❔       |
| OCR                | PP-OCRv3-tiny                | 2.4+10.7  | ✅       | ✅       | ✅       | --                                        | --                                      | --                       | --      |

## Community

<div id="fastdeploy-community"></div>

- If you have any question or suggestion, please give us your valuable input via GitHub Issues
- **Join Us👬：** Scan the QR code via WeChat to join our **FastDeploy technology communication group** （you ）

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/188544891-0ba025e5-61bd-425e-8097-8e982af9080e.jpeg"  width = "225" height = "288" />
</div>

## Acknowledge

<div id="fastdeploy-acknowledge"></div>

We sincerely appreciate  the open-sourced capabilities in [EasyEdge](https://ai.baidu.com/easyedge/app/openSource) as we adopt it for the SDK generation and download in this project.

## License

<div id="fastdeploy-license"></div>

FastDeploy is provided under [Apache-2.0](./LICENSE).
