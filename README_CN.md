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

**⚡️FastDeploy**是一款**易用高效**的推理部署开发套件。覆盖业界🔥**热门CV、NLP、Speech的AI模型**并提供📦**开箱即用**的部署体验，包括图像分类、目标检测、图像分割、人脸检测、人脸识别、人体关键点识别、文字识别、语义理解等多任务，满足开发者**多场景**，**多硬件**、**多平台**的产业部署需求。

|       [Object Detection](examples/vision)                                       | [3D Object Detection](https://github.com/PaddlePaddle/FastDeploy/issues/6)                                                                                             | [Semantic Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                     | [Potrait Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                                         |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188270227-1a4671b3-0123-46ab-8d0f-0e4132ae8ec0.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">                                                                                                                    | <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px">                                                                                                                                 |
| [**Image Matting**](examples/vision/matting)                 |  [**Real-Time Matting**](examples/vision/matting)           | [**OCR**](examples/vision/ocr)                  |[**Face Alignment**](examples/vision/ocr)
| <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px"> |<img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"  >                                                                                                                              |<img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px">  |
| [**Pose Estimation**](examples/vision/keypointdetection)                                                                                     | [**Behavior Recognition**](https://github.com/PaddlePaddle/FastDeploy/issues/6)                                                                                        |  [**NLP**](examples/text)                                                                                                                                                                                                           |[**Speech**](examples/audio/pp-tts)  
| <img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/200162475-f5d85d70-18fb-4930-8e7e-9ca065c1d618.gif' height="126px" width="190px">  |  <p align="left">**input** ：早上好，今天是2020<br>/10/29，最低温度是-3°C。<br><br> <p align="left">**output**: [<img src="https://user-images.githubusercontent.com/54695910/200161645-871e08da-5a31-4736-879c-a88bb171a676.png" width="170" style="max-width: 100%;">](https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/parakeet_espnet_fs2_pwg_demo/tn_g2p/parakeet/001.wav)</p>|


## 近期更新

- 🔥 **【直播分享】2022.11.09 20:30～21:30，《覆盖云边端全场景，150+热门模型快速部署》。微信扫码报名**
- 🔥 **【直播分享】2022.11.10 20:30～21:30，《瑞芯微、晶晨、恩智浦等10+AI硬件部署，直达产业落地》。微信扫码报名**
- 🔥 **【直播分享】2022.11.10 19:00～20:00，《10+热门模型在RK3588、RK3568部署实战》。微信扫码报名**
 <div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg" width = "120" height = "120" />
  </div>

- 🔥 **2022.10.31：Release FastDeploy [release v0.5.0](https://github.com/PaddlePaddle/FastDeploy/tree/release/0.5.0)**
    -  **🖥️ 服务端部署：支持推理速度更快的后端，支持更多的模型**
        -  集成 Paddle Inference TensorRT后端，并保证其使用与Paddle Inference、TensorRT、OpenVINO、ONNX Runtime、Paddle Lite等一致的开发体验；
        -  支持并测试 Graphcore IPU 通过 Paddle Inference后端;
        -  优化[一键模型量化工具](tools/quantization)，支持YOLOv7、YOLOv6、YOLOv5等视觉模型，在CPU和GPU推理速度可提升1.5～2倍；
        -  新增 [PP-Tracking](./examples/vision/tracking/pptracking) 和 [RobustVideoMatting](./examples/vision/matting) 等模型；

- 🔥 **2022.10.24：Release FastDeploy [release v0.4.0](https://github.com/PaddlePaddle/FastDeploy/tree/release/0.4.0)**
    -  **🖥️ 服务端部署：推理速度大升级**
        -  升级 GPU 端到端的优化，在YOLO系列上，模型推理速度从 43ms 提升到 25ms；
        -  新增 [TinyPose](examples/vision/keypointdetection/tiny_pose) and [PicoDetji lianTinyPose](examples/vision/keypointdetection/det_keypoint_unite)Pipeline部署能力；
    -  **📲 移动端和端侧部署：移动端后端能力升级，支持更多的CV模型**
       - 集成 Paddle Lite，并保证其使用与服务端常用推理引擎 Paddle Inference、TensorRT、OpenVINO、ONNX Runtime 等一致的开发体验；
       - 新增 [轻量化目标检测模型](examples/vision/detection/paddledetection/android)和[分类模型](examples/vision/classification/paddleclas/android)的安卓端部署能力；
    -  **<img src="https://user-images.githubusercontent.com/54695910/200179541-05f8e187-9f8b-444c-9252-d9ce3f1ab05f.png" width = "18" height = "18" /> Web和小程序部署：新增Web端部署能力**
       - 集成 Paddle.js部署能力，新增 OCR、目标检测、人像分割背景替换、物体识别等Web端部署能力和[Demo](examples/application/js)；

## 目录

* <details open> <summary><style="font-size:100px"><b>📖 文档教程（点击可收缩）</b></font></summary>
    
   - 安装文档
        - [预编译库下载安装](docs/cn/build_and_install/download_prebuilt_libraries.md)
        - [GPU部署环境编译安装](docs/cn/build_and_install/gpu.md)
        - [CPU部署环境编译安装](docs/cn/build_and_install/cpu.md)
        - [IPU部署环境编译安装](docs/cn/build_and_install/ipu.md)
        - [Jetson部署环境编译安装](docs/cn/build_and_install/jetson.md)
        - [Android平台部署环境编译安装](docs/cn/build_and_install/android.md)
   - 快速使用
        - [Python部署示例](docs/cn/quick_start/models/python.md)
        - [C++部署示例](docs/cn/quick_start/models/cpp.md)
        - [Runtime Python使用示例](docs/cn/quick_start/runtime/python.md)
        - [Runtime C++使用示例](docs/cn/quick_start/runtime/cpp.md)
   - API文档(进行中)
        - [Python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/)
        - [C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/)
   - 性能调优
        - [量化加速](docs/cn/quantize.md)
   - 常见问题
        - [1. 如何配置模型部署的推理后端](docs/cn/faq/how_to_change_backend.md)
        - [2. Windows上C++ SDK如何使用](docs/cn/faq/use_sdk_on_windows.md)
        - [3. Android上如何使用FastDeploy](docs/cn/faq/use_sdk_on_android.md)(进行中)
        - [4. TensorRT使用中的一些技巧](docs/cn/faq/tensorrt_tricks.md)
        - [5. 如何增加新的模型](docs/cn/faq/develop_a_new_model.md)(进行中)
   - 更多FastDeploy部署模块
        - [服务化部署](../serving)
        - [Benchmark测试](../benchmark)
</details>

* **🖥️ 服务器端部署**
    * [Python SDK快速开始](#fastdeploy-quick-start-python)  
    * [C++ SDK快速开始](#fastdeploy-quick-start-cpp)
    * [服务端模型支持列表](#fastdeploy-server-models)
* **📲 移动端和端侧部署**
    * [Paddle Lite NPU部署](#fastdeploy-edge-sdk-npu)
    * [端侧模型支持列表](#fastdeploy-edge-models)
* **<img src="https://user-images.githubusercontent.com/54695910/200179541-05f8e187-9f8b-444c-9252-d9ce3f1ab05f.png" width = "18" height = "18" /> Web和小程序部署**  
    * [Web端模型支持列表](#fastdeploy-web-models)
* [**社区交流**](#fastdeploy-community)
* [**Acknowledge**](#fastdeploy-acknowledge)  
* [**License**](#fastdeploy-license)

## 🖥️ 服务端部署

<div id="fastdeploy-quick-start-python"></div>

<details open> <summary><style="font-size:100px"><b>Python SDK快速开始（点击可收缩）</b></font></summary>

#### 快速安装

##### 前置依赖
- CUDA >= 11.2、cuDNN >= 8.0、Python >= 3.6
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

</details>

<div id="fastdeploy-quick-start-cpp"></div>

<details>
<summary><style="font-size:100px"><b>C++ SDK快速开始（点开查看详情）</b></font></summary>
    

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

### 服务端模型支持列表 🔥🔥🔥🔥🔥

<div id="fastdeploy-server-models"></div>

符号说明: (1)  ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持; <br>
链接说明：「模型列」会跳转到模型推理Demo代码

| 任务场景                   | 模型                                                                                           | API                                                                                                                                       | Linux   | Linux      | Win     | Win        | Mac     | Mac     | Linux       | Linux         | Linux         | Linux   |
|:----------------------:|:--------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:----------:|:-------:|:----------:|:-------:|:-------:|:-----------:|:-------------:|:-------------:|:-------:|
| ---                    | ---                                                                                          | ---                                                                                                                                       | X86 CPU | NVIDIA GPU | X86 CPU | NVIDIA GPU | X86 CPU | Arm CPU | AArch64 CPU | NVIDIA Jetson | Graphcore IPU | Serving |
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


## 📲 移动端和端侧部署 🔥🔥🔥🔥

<div id="fastdeploy-edge-doc"></div>

### Paddle Lite NPU部署

<div id="fastdeploy-edge-sdk-npu"></div>

- [瑞芯微-NPU/晶晨-NPU/恩智浦-NPU](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)

### 📲 端侧模型支持列表

<div id="fastdeploy-edge-models"></div>

| 任务场景               | 模型                                                                                        | 大小(MB)   | Linux   | Android | iOS     | Linux                      | Linux                                | Linux                             | Linux                    | 更新中...  |
|:------------------:|:-----------------------------------------------------------------------------------------:|:--------:|:-------:|:-------:|:-------: |:------------------:|:------------------------------------:|:---------------------------------:|:------------------------:|:-------:|
| ---                | ---                                                                                       | ---      | ARM CPU | ARM CPU | ARM CPU | 瑞芯微NPU<br>RK3568/RK3588 | 瑞芯微NPU<br>RV1109/RV1126/RK1808 | 晶晨NPU <br>A311D/S905D/C308X | 恩智浦NPU<br>i.MX&nbsp;8M&nbsp;Plus | 更新中...｜ |
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
    
    
## <img src="https://user-images.githubusercontent.com/54695910/200179541-05f8e187-9f8b-444c-9252-d9ce3f1ab05f.png" width = "18" height = "18" /> Web和小程序部署

<div id="fastdeploy-web-models"></div>
    
| 任务场景               | 模型                                                                                          | [web_demo](examples/application/js/web_demo) |
|:------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ---                | ---                                                                                         | [Paddle.js](examples/application/js)         |
| Detection          | [FaceDetection](examples/application/js/web_demo/src/pages/cv/detection)                    | ✅                                            |
| Detection          | [ScrewDetection](examples/application/js/web_demo/src/pages/cv/detection)                   | ✅                                            |
| Segmentation       | [PaddleSeg/HumanSeg](./examples/application/js/web_demo/src/pages/cv/segmentation/HumanSeg) | ✅                                            |
| Object Recognition | [GestureRecognition](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| Object Recognition | [ItemIdentification](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| OCR                | [PaddleOCR/PP-OCRv3](./examples/application/js/web_demo/src/pages/cv/ocr)                   | ✅                                            |
 
## 社区交流

<div id="fastdeploy-community"></div>

- **加入社区👬：** 微信扫描二维码，进入**FastDeploy技术交流群**

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg"  width = "225" height = "225" />
</div>

## Acknowledge

<div id="fastdeploy-acknowledge"></div>

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，在此表示感谢。

## License

<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
