English | [简体中文](README_CN.md)

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



**⚡️FastDeploy** is an **accessible and efficient** deployment Development Toolkit. It covers 🔥**critical CV、NLP、Speech AI models** in the industry and provides 📦**out-of-the-box** deployment experience. It covers image classification, object detection, image segmentation, face detection, face recognition, human keypoint detection, OCR, semantic understanding and other tasks to meet developers' industrial deployment needs for **multi-scenario**, **multi-hardware** and **multi-platform** .

|       [Object Detection](examples/vision)                                       | [3D Object Detection](https://github.com/PaddlePaddle/FastDeploy/issues/6)                                                                                             | [Semantic Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                     | [Potrait Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                                         |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188270227-1a4671b3-0123-46ab-8d0f-0e4132ae8ec0.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">                                                                                                                    | <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px">                                                                                                                                 |
| [**Image Matting**](examples/vision/matting)                 |  [**Real-Time Matting**](examples/vision/matting)           | [**OCR**](examples/vision/ocr)                  |[**Face Alignment**](examples/vision/ocr)
| <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px"> |<img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"  >                                                                                                                              |<img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px">  |
| [**Pose Estimation**](examples/vision/keypointdetection)                                                                                     | [**Behavior Recognition**](https://github.com/PaddlePaddle/FastDeploy/issues/6)                                                                                        |  [**NLP**](examples/text)                                                                                                                                                                                                           |[**Speech**](examples/audio/pp-tts)  
| <img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/200162475-f5d85d70-18fb-4930-8e7e-9ca065c1d618.gif' height="126px" width="190px">  |  <p align="left">**input** ：Life was like a box of chocolates, you never know what you're gonna get.<br> <p align="left">**output**: [<img src="https://user-images.githubusercontent.com/54695910/200161645-871e08da-5a31-4736-879c-a88bb171a676.png" width="170" style="max-width: 100%;">](https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_1.wav)</p>|


## 📣 Recent Updates

- 🔥 **【Live Preview】2022.11.09 20:30～21:30，《Covering the full spectrum of cloud-side scenarios with 150+ popular models for rapid deployment》**
- 🔥 **【Live Preview】2022.11.10 20:30～21:30，《10+ AI hardware deployments from Rockchip, Amlogic, NXP and others, straight to industry landing》**
- 🔥 **【Live Preview】2022.11.10 19:00～20:00，《10+ popular models deployed in RK3588, RK3568 in action》**
    - Scan the QR code below using WeChat, follow the PaddlePaddle official account and fill out the questionnaire to join the WeChat group  
 <div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg" width = "120" height = "120" />
  </div>
  

- 🔥 **2022.10.31：Release FastDeploy [release v0.5.0](https://github.com/PaddlePaddle/FastDeploy/tree/release/0.5.0)** <br>
    -  **🖥️ Data Center and Cloud Deployment: Support more backend, Support more CV models**
        -  Support Paddle Inference TensorRT, and provide a seamless deployment experience with other inference engines include Paddle Inference、Paddle Lite、TensorRT、OpenVINO、ONNX Runtime；
        -  Support Graphcore IPU through paddle Inference;
        -  Support tracking model [PP-Tracking](./examples/vision/tracking/pptracking) and [RobustVideoMatting](./examples/vision/matting) model；
        -  Support [one-click model quantization](tools/quantization) to improve model inference speed by 1.5 to 2 times on CPU & GPU platform. The supported quantized model are YOLOv7, YOLOv6, YOLOv5, etc. 

- 🔥 **2022.10.24：Release FastDeploy [release v0.4.0](https://github.com/PaddlePaddle/FastDeploy/tree/release/0.4.0)** <br>
    -  **🖥️ Data Center and Cloud Deployment: end-to-end optimization, Support more CV and NLP model**
       - end-to-end optimization on GPU, [YOLO series](examples/vision/detection) model end-to-end inference speedup from 43ms to 25ms;
       - Support CV models include PP-OCRv3, PP-OCRv2, PP-TinyPose, PP-Matting, etc. and provides [end-to-end deployment demos](examples/vision/detection/);
       - Support information extraction model is UIE, and provides [end-to-end deployment demos](examples/text/uie);
       - Support [TinyPose](examples/vision/keypointdetection/tiny_pose) and [PicoDet and TinyPose](examples/vision/keypointdetection/det_keypoint_unite)Pipeline deployment.
    -  **📲 Mobile and Edge Device Deployment: support new backend，support more CV model**
       - Integrate Paddle Lite and provide a seamless deployment experience with other inference engines include TensorRT、OpenVINO、ONNX Runtime、Paddle Inference；
       - Support [Lightweight Detection Model](examples/vision/detection/paddledetection/android) and [classification model](examples/vision/classification/paddleclas/android) on Android Platform，Download to try it out.
    -  **<img src="https://user-images.githubusercontent.com/54695910/200179541-05f8e187-9f8b-444c-9252-d9ce3f1ab05f.png" width = "18" height = "18" />Web-Side Deployment: support more CV model**  
       - Web deployment and Mini Program deployment New [OCR and other CV models](examples/application/js) capability.
      

## Contents

* <details open> <summary><style="font-size:100px"><b>📖 Tutorials（click to shrink） </b></font></summary>
    
   - Install
        - [How to Install FastDeploy Prebuilt Libraries](en/build_and_install/download_prebuilt_libraries.md)
        - [How to Build and Install FastDeploy Library on GPU Platform](en/build_and_install/gpu.md)
        - [How to Build and Install FastDeploy Library on CPU Platform](en/build_and_install/cpu.md)
        - [How to Build and Install FastDeploy Library on IPU Platform](en/build_and_install/ipu.md)
        - [How to Build and Install FastDeploy Library on  Nvidia Jetson Platform](en/build_and_install/jetson.md)
        - [How to Build and Install FastDeploy Library on Android Platform](en/build_and_install/android.md)
   - A Quick Start - Demos
        - [Python Deployment Demo](en/quick_start/models/python.md)
        - [C++ Deployment Demo](en/quick_start/models/cpp.md)
        - [A Quick Start on Runtime Python](en/quick_start/runtime/python.md)
        - [A Quick Start on Runtime C++](en/quick_start/runtime/cpp.md)
   - API (To be continued)
        - [Python API](https://baidu-paddle.github.io/fastdeploy-api/python/html/)
        - [C++ API](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/)
   - Performance Optimization
        - [Quantization Acceleration](en/quantize.md)
   - Frequent Q&As
        - [1. How to Change Inference Backends](en/faq/how_to_change_backend.md)
        - [2. How to Use FastDeploy C++ SDK on Windows Platform](en/faq/use_sdk_on_windows.md)
        - [3. How to Use FastDeploy C++ SDK on Android Platform](en/faq/use_sdk_on_android.md)(To be Continued)
        - [4. Tricks of TensorRT](en/faq/tensorrt_tricks.md)
        - [5. How to Develop a New Model](en/faq/develop_a_new_model.md)(To be Continued)
   - More FastDeploy Deployment Module
        - [deployment AI Model as a Service](../serving)
        - [Benchmark Testing](../benchmark)
</details>

* **🖥️ Data Center and Cloud Deployment**
  * [A Quick Start for Python SDK](#fastdeploy-quick-start-python)  
  * [A Quick Start for C++ SDK](#fastdeploy-quick-start-cpp)
  * [Supported Data Center and Cloud Model List](#fastdeploy-server-models)
* **📲 Mobile and Edge Device Deployment**
  * [Paddle Lite NPU Deployment](#fastdeploy-edge-sdk-npu)
  * [Supported Mobile and Edge Model List](#fastdeploy-edge-models)
* **<img src="https://user-images.githubusercontent.com/54695910/200179541-05f8e187-9f8b-444c-9252-d9ce3f1ab05f.png" width = "18" height = "18" />Web and Mini Program Deployment** 
  * [Supported Web and Mini Program Model List](#fastdeploy-web-models)
* [**Community**](#fastdeploy-community)
* [**Acknowledge**](#fastdeploy-acknowledge)  
* [**License**](#fastdeploy-license)

## 🖥️ Data Center and Cloud Deployment

<div id="fastdeploy-quick-start-python"></div>

<details open>
<summary><style="font-size:100px"><b>A Quick Start for Python SDK（click to shrink）</b></font></summary>


#### Installation

##### Prerequisites

- CUDA >= 11.2 、cuDNN >= 8.0  、 Python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

##### Install Fastdeploy SDK with CPU&GPU support

```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

##### [Conda Installation (Recommended)](docs/cn/build_and_install/download_prebuilt_libraries.md)

```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

##### Install Fastdeploy SDK with only CPU support

```bash
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### Python Inference Example

* Prepare models and pictures

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference results

```python
# For deployment of GPU/TensorRT, please refer to examples/vision/detection/paddledetection/python
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
</details>
         
<div id="fastdeploy-quick-start-cpp"></div>

<details>
<summary><style="font-size:100px"><b>A Quick Start for C++ SDK（click to expand）</b></font></summary>

#### Installation

- Please refer to [C++ Prebuilt Libraries Download](docs/cn/build_and_install/download_prebuilt_libraries.md)

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
  model.Predict(&im, &res);

  auto vis_im = vision::Visualize::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
 }
```
</details>

For more deployment models, please refer to [Vision Model Deployment Examples](examples/vision) .


<div id="fastdeploy-server-models"></div>

### Supported Data Center and Web Model List🔥🔥🔥🔥🔥

Notes: ✅: already supported; ❔: to be supported in the future;  N/A: Not Available;

<div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/198620704-741523c1-dec7-44e5-9f2b-29ddd9997344.png" />
</div>
  

| Task                          | Model                                                                                   | API                                                                                                                               | Linux                 | Linux                    | Win                      | Win                      | Mac                     | Mac                   | Linux                      | Linux                       | Linux                       |  Linux        |
|:-----------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------:|:---------------------:|:--------------------------:|:---------------------------:|:--------------------------:|:---------------------------:|
| ---                           | ---                                                                                     | ---                                                                                                                               | <font size=2> X86 CPU | <font size=2> NVIDIA GPU | <font size=2> Intel  CPU | <font size=2> NVIDIA GPU | <font size=2> Intel CPU | <font size=2> Arm CPU | <font size=2>  AArch64 CPU | <font size=2> NVIDIA Jetson | <font size=2> Graphcore IPU | Serving|
| Classification         | [PaddleClas/ResNet50](./examples/vision/classification/paddleclas)                           | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [TorchVison/ResNet](examples/vision/classification/resnet)                                   | [Python](./examples/vision/classification/resnet/python)/[C++](./examples/vision/classification/resnet/python/cpp)                        | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Classification         | [ltralytics/YOLOv5Cls](examples/vision/classification/yolov5cls)                             | [Python](./examples/vision/classification/yolov5cls/python)/[C++](./examples/vision/classification/yolov5cls/cpp)                         | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Classification         | [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas)                           | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas)                         | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas)                       | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/GhostNet](./examples/vision/classification/paddleclas)                           | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas)                        | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas)                        | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas)                        | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas)                       | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas)                    | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas)                        | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Classification         | [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas)                           | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ✅             | ❔       |
| Classification         | [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas)                    | [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/PP-YOLOE](./examples/vision/detection/paddledetection)                      | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection)                       | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection)                         | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection)                        | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection)                       | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection)                     | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/Faster-RCNN](./examples/vision/detection/paddledetection)                   | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [PaddleDetection/Mask-RCNN](./examples/vision/detection/paddledetection)                     | [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp)                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox)                              | [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp)                                           | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7)                                      | [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp)                                         | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [WongKinYiu/YOLOv7end2end_trt](./examples/vision/detection/yolov7end2end_trt)                | [Python](./examples/vision/detection/yolov7end2end_ort/python)/[C++](./examples/vision/detection/yolov7end2end_ort/cpp)                   | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             |               |         |
| Detection              | [WongKinYiu/YOLOv7end2end_ort_](./examples/vision/detection/yolov7end2end_ort)               | [Python](./examples/vision/detection/yolov7end2end_ort/python)/[C++](./examples/vision/detection/yolov7end2end_ort/cpp)                   | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             |               |         |
| Detection              | [meituan/YOLOv6](./examples/vision/detection/yolov6)                                         | [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp)                                         | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [ultralytics/YOLOv5](./examples/vision/detection/yolov5)                                     | [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp)                                         | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [WongKinYiu/YOLOR](./examples/vision/detection/yolor)                                        | [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp)                                           | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4)                          | [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite)                                   | [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp)                                 | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Detection              | [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus)                             | [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| KeyPoint               | [PaddleDetection/TinyPose](./examples/vision/keypointdetection/tiny_pose)                    | [Python](./examples/vision/keypointdetection/tiny_pose/python)/[C++](./examples/vision/keypointdetection/tiny_pose/python/cpp)            | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| KeyPoint               | [PaddleDetection/PicoDet + TinyPose](./examples/vision/keypointdetection/det_keypoint_unite) | [Python](./examples/vision/keypointdetection/det_keypoint_unite/python)/[C++](./examples/vision/keypointdetection/det_keypoint_unite/cpp) | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| HeadPose               | [omasaht/headpose](examples/vision/headpose)                                                 | [Python](./xamples/vision/headpose/fsanet/python)/[C++](./xamples/vision/headpose/fsanet/cpp/cpp)                                         | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Tracking               | [PaddleDetection/PP-Tracking](examples/vision/tracking/pptracking)                           | [Python](examples/vision/tracking/pptracking/python)/[C++](examples/vision/tracking/pptracking/cpp)                                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| OCR                    | [PaddleOCR/PP-OCRv2](./examples/vision/ocr)                                                  | [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/ocr/PP-OCRv3/cpp)                                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| OCR                    | [PaddleOCR/PP-OCRv3](./examples/vision/ocr)                                                  | [Python](./examples/vision/ocr/PP-OCRv3/python)/[C++](./examples/vision/ocr/PP-OCRv3/cpp)                                                 | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Segmentation           | [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg)                             | [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Segmentation           | [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg)                        | [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Segmentation           | [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg)                                  | [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Segmentation           | [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg)                      | [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Segmentation           | [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg)                                   | [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Segmentation           | [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg)                              | [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp)                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceDetection          | [biubug6/RetinaFace](./examples/vision/facedet/retinaface)                                   | [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp)                                     | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceDetection          | [Linzaer/UltraFace](./examples/vision/facedet/ultraface)                                     | [ Python](./examples/vision/facedet/ultraface/python)/[C++](./examples/vision/facedet/ultraface/cpp)                                      | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceDetection          | [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face)                                | [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp)                                     | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ✅       |
| FaceDetection          | [insightface/SCRFD](./examples/vision/facedet/scrfd)                                         | [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp)                                               | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceAlign              | [Hsintao/PFLD](examples/vision/facealign/pfld)                                               | [Python](./examples/vision/facealign/pfld/python)/[C++](./examples/vision/facealign/pfld/cpp)                                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceRecognition        | [insightface/ArcFace](./examples/vision/faceid/insightface)                                  | [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)                                     | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceRecognition        | [insightface/CosFace](./examples/vision/faceid/insightface)                                  | [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)                                     | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceRecognition        | [insightface/PartialFC](./examples/vision/faceid/insightface)                                | [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)                                     | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| FaceRecognition        | [insightface/VPL](./examples/vision/faceid/insightface)                                      | [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp)                                     | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Matting                | [ZHKKKe/MODNet](./examples/vision/matting/modnet)                                            | [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp)                                             | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Matting                | [PeterL1n/RobustVideoMatting]()                                                              | [Python](./examples/vision/matting/rvm/python)/[C++](./examples/vision/matting/rvm/cpp)                                                   | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Matting                | [PaddleSeg/PP-Matting](./examples/vision/matting/ppmatting)                                  | [Python](./examples/vision/matting/ppmatting/python)/[C++](./examples/vision/matting/ppmatting/cpp)                                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Matting                | [PaddleSeg/PP-HumanMatting](./examples/vision/matting/modnet)                                | [Python](./examples/vision/matting/ppmatting/python)/[C++](./examples/vision/matting/ppmatting/cpp)                                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Matting                | [PaddleSeg/ModNet](./examples/vision/matting/modnet)                                         | [Python](./examples/vision/matting/ppmatting/python)/[C++](./examples/vision/matting/ppmatting/cpp)                                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| Information Extraction | [PaddleNLP/UIE](./examples/text/uie)                                                         | [Python](./examples/text/uie/python)/[C++](./examples/text/uie/cpp)                                                                       | ✅       | ✅          | ✅       | ✅          | ✅       | ✅       | ✅           | ✅             | ❔             | ❔       |
| NLP                    | [PaddleNLP/ERNIE-3.0](./examples/text/ernie-3.0)                                             | Python/C++                                                                                                                                | ❔       | ❔          | ❔       | ❔          | ❔       | ❔       | ❔           | ❔             | ❔             | ✅       |
| Speech                 | [PaddleSpeech/PP-TTS](./examples/text/uie)                                                   | [Python](examples/audio/pp-tts/python)/C++                                                                                                | ❔       | ❔          | ❔       | ❔          | ❔       | ❔       | ❔           | ❔             | --            | ✅       |
    

<div id="fastdeploy-edge-doc"></div>
    
## 📲 Mobile and Edge Device Deployment 


<div id="fastdeploy-edge-sdk-npu"></div>
    
### Paddle Lite NPU Deployment

- [Rockchip-NPU / Amlogic-NPU / NXP-NPU](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)

<div id="fastdeploy-edge-models"></div>
    
### Supported Mobile and Edge Model List 🔥🔥🔥🔥

<div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/198620704-741523c1-dec7-44e5-9f2b-29ddd9997344.png" />
</div>

|  Task              | Model                        | Size (MB) | Linux   | Android | iOS     | Linux  |Linux                 | Linux                                   | Linux                    | TBD...  |
|:------------------:|:----------------------------:|:---------:|:-------:|:-------:|:-------:|:-----------------------------------------:|:---------------------------------------:|:------------------------:|:-------:|:-------:|
| ---                | ---                          | ---       | ARM CPU | ARM CPU | ARM CPU |Rockchip-NPU<br>RK3568/RK3588 |Rockchip-NPU<br>RV1109/RV1126/RK1808 | Amlogic-NPU <br>A311D/S905D/C308X | NXP-NPU<br>i.MX&nbsp;8M&nbsp;Plus | TBD...｜ |
| Classification     | [PaddleClas/PP-LCNet](examples/vision/classification/paddleclas)                          | 11.9     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/PP-LCNetv2](examples/vision/classification/paddleclas)                        | 26.6     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/EfficientNet](examples/vision/classification/paddleclas)                      | 31.4     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/GhostNet](examples/vision/classification/paddleclas)                          | 20.8     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/MobileNetV1](examples/vision/classification/paddleclas)                       | 17       | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/MobileNetV2](examples/vision/classification/paddleclas)                       | 14.2     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/MobileNetV3](examples/vision/classification/paddleclas)                       | 22       | ✅       | ✅       | ❔       | ❔                          | ❔                                    | ❔                                 | ❔                        | --      |
| Classification     | [PaddleClas/ShuffleNetV2](examples/vision/classification/paddleclas)                      | 9.2      | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/SqueezeNetV1.1](examples/vision/classification/paddleclas)                    | 5        | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/Inceptionv3](examples/vision/classification/paddleclas)                       | 95.5     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/PP-HGNet](examples/vision/classification/paddleclas)                          | 59       | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/SwinTransformer_224_win7](examples/vision/classification/paddleclas)          | 352.7    | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-PicoDet_s_320_coco](examples/vision/detection/paddledetection)        | 4.1      | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-PicoDet_s_320_lcnet](examples/vision/detection/paddledetection)       | 4.9      | ✅       | ✅       | ❔       | ❔                          | ✅                                    | ✅                                 | ✅                        | --      |
| Detection          | [PaddleDetection/CenterNet](examples/vision/detection/paddledetection)                    | 4.8      | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/YOLOv3_MobileNetV3](examples/vision/detection/paddledetection)           | 94.6     | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLO_tiny_650e_coco](examples/vision/detection/paddledetection)       | 4.4      | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/SSD_MobileNetV1_300_120e_voc](examples/vision/detection/paddledetection) | 23.3     | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLO_ResNet50vd](examples/vision/detection/paddledetection)           | 188.5    | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLOv2_ResNet50vd](examples/vision/detection/paddledetection)         | 218.7    | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLO_crn_l_300e_coco](examples/vision/detection/paddledetection)      | 209.1    | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | YOLOv5s                                                                                   | 29.3     | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Face Detection     | BlazeFace                                                                                 | 1.5      | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Face Detection     | RetinaFace                                                                                | 1.7      | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Keypoint Detection | [PaddleDetection/PP-TinyPose](examples/vision/keypointdetection/tiny_pose)                | 5.5      | ✅       | ❔       | ❔       | ❔                          | ❔                                    | ❔                                 | ❔                        | --      |
| Segmentation       | [PaddleSeg/PP-LiteSeg(STDC1)](examples/vision/segmentation/paddleseg)                     | 32.2     | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg-Lite](examples/vision/segmentation/paddleseg)                      | 0.556    | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/HRNet-w18](examples/vision/segmentation/paddleseg)                             | 38.7     | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg](examples/vision/segmentation/paddleseg)                           | 107.2    | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/Unet](examples/vision/segmentation/paddleseg)                                  | 53.7     | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/Deeplabv3](examples/vision/segmentation/paddleseg)                             | 150      | ❔       | ❔       | ❔       | ✅                          |                                      |                                   |                          |         |
| OCR                | PaddleOCR/PP-OCRv1                                                                        | 2.3+4.4  | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| OCR                | [PaddleOCR/PP-OCRv2](examples/vision/ocr/PP-OCRv2)                                        | 2.3+4.4  | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| OCR                | [PaddleOCR/PP-OCRv3](examples/vision/ocr/PP-OCRv3)                                        | 2.4+10.6 | ✅       | ❔       | ❔       | ❔                          | ❔                                    | ❔                                 | ❔                        | --      |
| OCR                | PaddleOCR/PP-OCRv3-tiny                                                                   | 2.4+10.7 | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |

    
## <img src="https://user-images.githubusercontent.com/54695910/200179541-05f8e187-9f8b-444c-9252-d9ce3f1ab05f.png" width = "18" height = "18" /> Web and Mini Program Deployment

<div id="fastdeploy-web-models"></div>
    
| Task                | Model                                                                                         | [web_demo](examples/application/js/web_demo) |
|:------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ---                | ---                                                                                         | [Paddle.js](examples/application/js)         |
| Detection          | [FaceDetection](examples/application/js/web_demo/src/pages/cv/detection)                    | ✅                                            |
| Detection          | [ScrewDetection](examples/application/js/web_demo/src/pages/cv/detection)                   | ✅                                            |
| Segmentation       | [PaddleSeg/HumanSeg](./examples/application/js/web_demo/src/pages/cv/segmentation/HumanSeg) | ✅                                            |
| Object Recognition | [GestureRecognition](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| Object Recognition | [ItemIdentification](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| OCR                | [PaddleOCR/PP-OCRv3](./examples/application/js/web_demo/src/pages/cv/ocr)                   | ✅                                            |

    
## Community

<div id="fastdeploy-community"></div>

- If you have any question or suggestion, please give us your valuable input via GitHub Issues
- **Join Us👬：**
    - **Slack**：Join our [Slack community](https://join.slack.com/t/fastdeployworkspace/shared_invite/zt-1hhvpb279-iw2pNPwrDaMBQ5OQhO3Siw) and chat with other community members about ideas
    - **WeChat**：join our **WeChat community** and chat with other community members about ideas

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg"  width = "225" height = "225" />
</div>



## Acknowledge

<div id="fastdeploy-acknowledge"></div>

We sincerely appreciate the open-sourced capabilities in [EasyEdge](https://ai.baidu.com/easyedge/app/openSource) as we adopt it for the SDK generation and download in this project.

## License

<div id="fastdeploy-license"></div>

FastDeploy is provided under the [Apache-2.0](./LICENSE).
