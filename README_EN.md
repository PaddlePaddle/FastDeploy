English | [简体中文](README_CN.md) | [हिन्दी](./docs/docs_i18n/README_हिन्दी.md) | [日本語](./docs/docs_i18n/README_日本語.md) | [한국인](./docs/docs_i18n/README_한국인.md) | [Pу́сский язы́к](./docs/docs_i18n/README_Pу́сский_язы́к.md)


![⚡️FastDeploy](https://user-images.githubusercontent.com/31974251/185771818-5d4423cd-c94c-4a49-9894-bc7a8d1c29d0.png)

</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/FastDeploy?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>
</p>
<p align="center">
    <a href="/docs/en/build_and_install"><b> Installation </b></a>
    |
    <a href="docs/README_EN.md"><b> Documents </b></a>
    | <a href="/README_EN.md#Quick-Start"><b> Quick Start </b></a> |
    <a href="https://baidu-paddle.github.io/fastdeploy-api/"><b> API Docs </b></a>
    |
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><b> Release Notes </b></a>
</p>
<div align="center">
    
[<img src='https://user-images.githubusercontent.com/54695910/200465949-da478e1b-21ce-43b8-9f3f-287460e786bd.png' height="80px" width="110px">](examples/vision/classification)
[<img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="80px" width="110px">](examples/vision/detection)
[<img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="80px" width="110px">](examples/vision/segmentation/paddleseg)
[<img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="80px" width="110px">](examples/vision/segmentation/paddleseg)
[<img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="80px" width="110px">](examples/vision/matting)
[<img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="80px" width="110px">](examples/vision/matting)
[<img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="80px" width="110px">](examples/vision/ocr)<br>
[<img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="80px" width="110px">](examples/vision/facealign)
[<img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="80px" width="110px">](examples/vision/keypointdetection)
[<img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="80px" width="110px">](https://user-images.githubusercontent.com/54695910/200162475-f5d85d70-18fb-4930-8e7e-9ca065c1d618.gif)
[<img src='https://user-images.githubusercontent.com/54695910/200162475-f5d85d70-18fb-4930-8e7e-9ca065c1d618.gif' height="80px" width="110px">](examples/text)
[<img src='https://user-images.githubusercontent.com/54695910/212314909-77624bdd-1d12-4431-9cca-7a944ec705d3.png' height="80px" width="110px">](https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/parakeet_espnet_fs2_pwg_demo/tn_g2p/parakeet/001.wav)
</div>

**⚡️FastDeploy** is an **Easy-to-use** and **High Performance** AI model deployment toolkit for Cloud, Mobile and Edge with 📦**out-of-the-box and unified experience**, 🔚**end-to-end optimization** for over **🔥160+ Text, Vision, Speech and Cross-modal AI models**.
Including [image classification](examples/vision/classification), [object detection](examples/vision/detection), [OCR](./examples/vision/ocr), [face detection](./examples/vision/facedet), [matting](./examples/vision/matting), [pp-tracking](./examples/vision/tracking/pptracking), [NLP](./examples/text), [stable difussion](./examples/multimodal/stable_diffusion), [TTS](./examples/audio/pp-tts) and other tasks to meet developers' industrial deployment needs for **multi-scenario**, **multi-hardware** and **multi-platform**.

<div align="center">
    
<img src="https://user-images.githubusercontent.com/54695910/213087724-7175953a-0e07-4af8-a4a1-5304163da2e0.png" >
    
</div>



##  🌠 Recent updates
- ✨✨✨ In **2023.01.17** we released [**YOLOv8**](./examples/vision/detection/paddledetection/) for deployment on FastDeploy series hardware, which includes [**Paddle YOLOv8**](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8) and [**ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics)

    - You can deploy [**Paddle YOLOv8**](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8) on [**Intel CPU**](./examples/vision/detection/paddledetection/python/infer_yolov8.py), [**NVIDIA GPU**](./examples/vision/detection/paddledetection/python/infer_yolov8.py), [**Jetson**](./examples/vision/detection/paddledetection/python/infer_yolov8.py), [**Phytium**](./examples/vision/detection/paddledetection/python/infer_yolov8.py), [**Kunlunxin**](./examples/vision/detection/paddledetection/python/infer_yolov8.py), [**HUAWEI Ascend**](./examples/vision/detection/paddledetection/python/infer_yolov8.py) ,[**ARM CPU**](./examples/vision/detection/paddledetection/cpp/infer_yolov8.cc)  [**RK3588**](./examples/vision/detection/paddledetection/rknpu2) and [**Sophgo TPU**](./examples/vision/detection/paddledetection/sophgo). Both **Python** deployments and **C++** deployments are included. 
    - You can deploy [**ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics) on [**Intel CPU**](./examples/vision/detection/yolov8), [**NVIDIA GPU**](./examples/vision/detection/yolov8), [**Jetson**](./examples/vision/detection/yolov8). Both **Python** deployments and **C++** deployments are included
    -  Fastdeploy supports quick deployment of multiple models, including **YOLOv8**, **PP-YOLOE+**, **YOLOv5** and other models
-  Serving deployment combined with VisualDL supports visual deployment. After the VDL service is started in the FastDeploy container, you can modify the model configuration, start/manage the model service, view performance data, and send requests on the VDL interface. For details, see related documents
    - [Serving deployment visualization](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/EN/vdl_management-en.md) 
    - [Serving request visualization](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/EN/client-en.md#use-visualdl-as-fastdeploy-client-for-request-visualization)
  
- **✨👥✨ Community**
  
  - **Slack**：Join our [Slack community](https://join.slack.com/t/fastdeployworkspace/shared_invite/zt-1m88mytoi-mBdMYcnTF~9LCKSOKXd6Tg) and chat with other community members about ideas
  - **Wechat**：Scan the QR code below using WeChat, follow the PaddlePaddle official account and fill out the questionnaire to join the WeChat group, and share the deployment industry implementation pain points with the community developers

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/207262688-4225bc39-4337-4966-a5cc-26bd6557d226.jpg"  width = "150" height = "150" />
</div>

## 🌌 Inference Backend and Abilities

<font size=0.5em>

|  | <img src="https://user-images.githubusercontent.com/54695910/213093175-052c3e47-75dc-4be8-9be9-6565532efa1c.png" width = "60" height = "40" />  | <img src="https://user-images.githubusercontent.com/54695910/213093173-27847120-bbb0-47b0-947f-8cf87142ed52.png" width = "75" height = "50"  /> |<img src="https://user-images.githubusercontent.com/54695910/213096791-8b47c875-6c89-4e1d-8c67-e226636844e1.png" width = "85" height = "60" />| <img src="https://user-images.githubusercontent.com/54695910/212475826-f52b0ef3-e512-49fe-9b52-e1b9d1e8b6c2.png" height = "30" />  | <img src="https://user-images.githubusercontent.com/54695910/212475825-9686ae78-bad9-4be9-852e-6ad23be209da.png" height = "30" />  | <img src="https://user-images.githubusercontent.com/54695910/212475822-067349d2-8c4a-4431-bf02-05387e2962a8.png" height = "30" />  |<img src="https://user-images.githubusercontent.com/54695910/212475820-5210efe0-3e9a-429a-ad9d-48e8da2ffd0b.png" height = "30" /> |
|:----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| X86_64&nbsp;CPU  |     |&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>&nbsp;&nbsp;&nbsp;    |   <img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>  | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>   |   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>   |
| NVDIA&nbsp;GPU    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    | <img src="https://user-images.githubusercontent.com/54695910/212474106-a297aa0d-9225-458e-b5b7-e31aec7cfa79.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>  |     |    |
|Phytium CPU  |    |     |  <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>  | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473393-ae1958bd-ab7d-4863-94b9-32863e600ba1.svg" height = "17"/>   |    |    |   |
| KunlunXin XPU |    |    | <img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>    |     |   |    |
| Huawei Ascend NPU |     |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>| <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>    |   |     |    |
|Graphcore&nbsp;IPU   |    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    |    |  <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/>  |    |    |  |
| Sophgo    |     |     |     | <img src="https://user-images.githubusercontent.com/54695910/212473382-e3e9063f-c298-4b61-ad35-a114aa6e6555.svg" height = "17"/>   |    |  |    |
|Intel graphics card  |     |     |     | <img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/>   |    |   | |
| Jetson    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    |<img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474106-a297aa0d-9225-458e-b5b7-e31aec7cfa79.svg" height = "17"/>   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>  |     |    |
|ARM&nbsp;CPU |    |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>| <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473393-ae1958bd-ab7d-4863-94b9-32863e600ba1.svg" height = "17"/>   |    | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>  |  <img src="https://user-images.githubusercontent.com/54695910/212473393-ae1958bd-ab7d-4863-94b9-32863e600ba1.svg" height = "17"/>  |
|RK3588 etc. |   |    | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473387-2559cc2a-024b-4452-806c-6105d8eb2339.svg" height = "17"/>  |    |    |    |
|RV1126 etc. |    |    | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>    |     |     |    |
| Amlogic |   |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>   |     |     |   |
|  NXP |   |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>   |     |    |    |
</font>

## 🔮 Contents
-  [✴️ A Quick Start for Python SDK](#fastdeploy-quick-start-python)  
-  [✴️ A Quick Start for C++ SDK](#fastdeploy-quick-start-cpp)
- **Installation**
    - [How to Install Prebuilt Library](docs/en/build_and_install/download_prebuilt_libraries.md)
    - [How to Build GPU Deployment Environment](docs/en/build_and_install/gpu.md)
    - [How to Build CPU Deployment Environment](docs/en/build_and_install/cpu.md)
    - [How to Build IPU Deployment Environment](docs/en/build_and_install/ipu.md)
    - [How to Build KunlunXin XPU Deployment Environment](docs/en/build_and_install/kunlunxin.md)
    - [How to Build RV1126 Deployment Environment](docs/en/build_and_install/rv1126.md)
    - [How to Build RKNPU2 Deployment Environment](docs/en/build_and_install/rknpu2.md)
    - [How to Build A311D Deployment Environment](docs/en/build_and_install/a311d.md)
    - [How to build Huawei Ascend Deployment Environment](docs/en/build_and_install/huawei_ascend.md)
    - [How to Build FastDeploy Library on Nvidia Jetson Platform](docs/en/build_and_install/jetson.md)
    - [How to Build FastDeploy Android C++ SDK](docs/en/build_and_install/android.md)
- **Quick Start**
    - [PP-YOLOE Python Deployment Example](docs/en/quick_start/models/python.md)
    - [PP-YOLOE C++ Deployment Example](docs/en/quick_start/models/cpp.md)
- **Demos on Different Backends**
    - [Runtime Python Inference](docs/en/quick_start/runtime/python.md)
    - [Runtime C++ Inference](docs/en/quick_start/runtime/cpp.md)
    - [How to Change Model Inference Backend](docs/en/faq/how_to_change_backend.md)
- **Serving Deployment**
    - [FastDeploy Serving Deployment Image Compilation](serving/docs/EN/compile-en.md)
    - [Serving Deployment](serving)
- **API Documents**
    - [Python API](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/)
    - [C++ API](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/)
    - [Android Java API](java/android)
- **Performance Tune-up**
    - [Quantization Acceleration](docs/en/quantize.md)
    - [Multi thread](/tutorials/multi_thread)
- **FAQ**
    - [1. Using the FastDeploy C++ SDK on Windows Platform](docs/en/faq/use_sdk_on_windows.md)
    - [2. FastDeploy to deploy on Android Platform](docs/en/faq/use_cpp_sdk_on_android.md)
    - [3. TensorRT Q&As](docs/en/faq/tensorrt_tricks.md)
- **More FastDeploy Deploy Modules**
    - [Benchmark Testing](benchmark)
- **Model list** 
  - [🖥️ Supported Server-side and Cloud Model List](#fastdeploy-server-models)
  - [📳 Supported Mobile and Edge Model List](#fastdeploy-edge-models)
  - [⚛️ Supported Web and Mini Program Model List](#fastdeploy-web-models)
- **💕 Developer Contributions**
    - [Develop a new model](docs/en/faq/develop_a_new_model.md)


## Quick Start💨

<div id="fastdeploy-quick-start-python"></div>

<details Open>
<summary><b>A Quick Start for Python SDK(click to fold)</b></summary><div>

#### 🎆 Installation

##### 🔸 Prerequisites

- CUDA >= 11.2 、cuDNN >= 8.0  、 Python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

##### 🔸 Install FastDeploy SDK with both CPU and GPU support

```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

##### [🔸 Conda Installation (Recommended✨)](docs/en/build_and_install/download_prebuilt_libraries.md)

```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

##### 🔸 Install FastDeploy SDK with only CPU support

```bash
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### 🎇 Python Inference Example

* Prepare model and picture

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

im = cv2.imread("000000014439.jpg")
model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")

result = model.predict(im)
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```

</div></details>

<div id="fastdeploy-quick-start-cpp"></div>

<details>
<summary><b>A Quick Start for C++ SDK(click to expand)</b></summary><div>

#### 🎆 Installation

- Please refer to [C++ Prebuilt Libraries Download](docs/en/build_and_install/download_prebuilt_libraries.md)

#### 🎇 C++ Inference Example

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
  auto im = cv::imread("000000014439.jpg");
  auto model = vision::detection::PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                          "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                          "ppyoloe_crn_l_300e_coco/infer_cfg.yml");

  vision::DetectionResult res;
  model.Predict(&im, &res);

  auto vis_im = vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
 }
```

</div></details>

For more deployment models, please refer to [Vision Model Deployment Examples](examples/vision) .

<div id="fastdeploy-server-models"></div>

## ✴️ ✴️ Server-side and Cloud Model List ✴️ ✴️

Notes: ✅: already supported; ❔: to be supported in the future;  N/A: Not Available;

<details open><summary><b> Server-side and cloud model list(click to fold)</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/115439700/212801271-5621419f-3997-4f00-94d5-63c8b6474aa8.png" height = "40"/>
</div>

| Task                   | Model                                                                                           | Linux                                            | Linux      | Win     | Win        | Mac     | Mac     | Linux       | Linux           | Linux         | Linux         | Linux   | Linux   | Linux   |
|:----------------------:|:--------------------------------------------------------------------------------------------:|:------------------------------------------------:|:----------:|:-------:|:----------:|:-------:|:-------:|:-----------:|:---------------:|:-------------:|:-------------:|:-------:|:-------:|:-------:|
| ---                    | ---                                                                                          | X86 CPU                                          | NVIDIA GPU | X86 CPU | NVIDIA GPU | X86 CPU | Arm CPU | AArch64 CPU | Phytium D2000 aarch64 | [NVIDIA Jetson](./docs/en/build_and_install/jetson.md) | [Graphcore IPU](./docs/en/build_and_install/ipu.md) | [kunlunxin XPU](./docs/en/build_and_install/kunlunxin.md) |[Huawei Ascend](./docs/en/build_and_install/huawei_ascend.md) |  [Serving](./serving) |
| Classification         | [PaddleClas/ResNet50](./examples/vision/classification/paddleclas)                           | [✅](./examples/vision/classification/paddleclas) | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       | ✅       |✅       |
| Classification         | [TorchVison/ResNet](examples/vision/classification/resnet)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ❔       |
| Classification         | [ltralytics/YOLOv5Cls](examples/vision/classification/yolov5cls)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ❔       |❔       |
| Classification         | [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           |  ✅       |✅       | ✅       |
| Classification         | [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas)                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/GhostNet](./examples/vision/classification/paddleclas)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas)                    | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ✅       |
| Classification         | [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       |✅       | ✅       |
| Detection              | 🔥🔥[PaddleDetection/PP-YOLOE+](./examples/vision/detection/paddledetection)                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ✅       |
| Detection              | [🔥PaddleDetection/YOLOv8](./examples/vision/detection/paddledetection)                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ❔      |
| Detection              | [🔥ultralytics/YOLOv8](./examples/vision/detection/yolov8)                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔               | ✅                           | ❔                           | ❔      |❔       | ❔      |
| Detection              | [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ❔       | ✅       |
| Detection              | [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection)                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅               | ✅                           | ❔                           | ✅       | ✅       | ✅       |
| Detection              | [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ✅       | ✅       |
| Detection              | [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ✅       | ✅       |
| Detection              | [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection)                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ✅       | ✅       |
| Detection              | [PaddleDetection/Faster-RCNN](./examples/vision/detection/paddledetection)                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |❔        | ✅       |
| Detection              | [PaddleDetection/Mask-RCNN](./examples/vision/detection/paddledetection)                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |❔        | ✅       |
| Detection              | [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox)                              | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ❔       |
| Detection              | [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7)                                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ❔       |
| Detection              | [WongKinYiu/YOLOv7end2end_trt](./examples/vision/detection/yolov7end2end_trt)                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ❔       |❔       | ❔       |
| Detection              | [WongKinYiu/YOLOv7end2end_ort](./examples/vision/detection/yolov7end2end_ort)               | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ❔       |❔       | ❔       |
| Detection              | [meituan/YOLOv6](./examples/vision/detection/yolov6)                                         | ✅                                                | ✅                        | ✅                        |✅       |  ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅                          | ❔       |
| Detection              | [ultralytics/YOLOv5](./examples/vision/detection/yolov5)                                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ✅       |✅       |
| Detection              | [WongKinYiu/YOLOR](./examples/vision/detection/yolor)                                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ❔      | ✅       | ❔       |
| Detection              | [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4)                          | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ❔       |❔       | ❔       |
| Detection              | [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           |  ?       | ❔       |❔       |❔       |
| Detection              | [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ❔       |❔       | ❔       |
| KeyPoint               | [PaddleDetection/TinyPose](./examples/vision/keypointdetection/tiny_pose)                    | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅        |❔       | ❔       |
| KeyPoint               | [PaddleDetection/PicoDet + TinyPose](./examples/vision/keypointdetection/det_keypoint_unite) | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅        | ❔       |❔       |
| HeadPose               | [omasaht/headpose](examples/vision/headpose)                                                 | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ❔       | ❔       |❔       |
| Tracking               | [PaddleDetection/PP-Tracking](examples/vision/tracking/pptracking)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ❔       | ❔       |❔       |
| OCR                    | [PaddleOCR/PP-OCRv2](./examples/vision/ocr)                                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ✅              |✅       | ❔       |
| OCR                    | [PaddleOCR/PP-OCRv3](./examples/vision/ocr)                                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅      |✅        | ✅       |
| Segmentation           | [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ✅                 |❔         | ❔       |
| Segmentation           | [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ✅                     |✅        | ❔       |
| Segmentation           | [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ✅                | ✅        |❔       |
| Segmentation           | [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg)                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                           | ✅        | ✅        |❔       |
| Segmentation           | [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ✅               | ✅         | ✅        |❔       |
| Segmentation           | [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg)                              | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                     | ✅               | ✅        |❔       |
| FaceDetection          | [biubug6/RetinaFace](./examples/vision/facedet/retinaface)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                  | ❔       |  ❔       | ❔       |
| FaceDetection          | [Linzaer/UltraFace](./examples/vision/facedet/ultraface)                                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       |  ❔       |❔       |
| FaceDetection          | [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face)                                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       |  ❔       |❔       |
| FaceDetection          | [insightface/SCRFD](./examples/vision/facedet/scrfd)                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       |  ❔       |❔       |
| FaceAlign              | [Hsintao/PFLD](examples/vision/facealign/pfld)                                               | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    |  ❔       | ❔       |❔       |
| FaceAlign              | [Single430/FaceLandmark1000](./examples/vision/facealign/face_landmark_1000)                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔       | ❔       | ❔       |
| FaceAlign              | [jhb86253817/PIPNet](./examples/vision/facealign)                                            | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔       | ❔       |❔       |
| FaceRecognition        | [insightface/ArcFace](./examples/vision/faceid/insightface)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       |  ❔       |❔       |
| FaceRecognition        | [insightface/CosFace](./examples/vision/faceid/insightface)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       | ❔       |❔       |
| FaceRecognition        | [insightface/PartialFC](./examples/vision/faceid/insightface)                                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       | ❔       | ❔       |
| FaceRecognition        | [insightface/VPL](./examples/vision/faceid/insightface)                                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔       | ❔       | ❔       |
| Matting                | [ZHKKKe/MODNet](./examples/vision/matting/modnet)                                            | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    |  ❔       | ❔       |❔       |
| Matting                | [PeterL1n/RobustVideoMatting]()                                                              | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔       |  ❔       | ❔       |
| Matting                | [PaddleSeg/PP-Matting](./examples/vision/matting/ppmatting)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ✅           |✅           | ❔       |
| Matting                | [PaddleSeg/PP-HumanMatting](./examples/vision/matting/modnet)                                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                   | ✅       |✅           |❔       |
| Matting                | [PaddleSeg/ModNet](./examples/vision/matting/modnet)                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                    | ❔        |❔       |   ❔       |
| Video Super-Resolution | [PaddleGAN/BasicVSR](./)                                                                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔       | ❔       |❔       |
| Video Super-Resolution | [PaddleGAN/EDVR](./examples/vision/sr/edvr)                                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔       |❔       | ❔       |
| Video Super-Resolution | [PaddleGAN/PP-MSVSR](./examples/vision/sr/ppmsvsr)                                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔       |❔       | ❔       |
| Information Extraction | [PaddleNLP/UIE](./examples/text/uie)                                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ✅                           | ❔                    | ❔         |❔       |         |
| NLP                    | [PaddleNLP/ERNIE-3.0](./examples/text/ernie-3.0)                                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ❔                           | ❔                    | ✅       |❔       | ✅       |
| Speech                 | [PaddleSpeech/PP-TTS](./examples/audio/pp-tts)                                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                | ❔                           | --                          |❔       |❔       | ✅       |

</div></details>

<div id="fastdeploy-edge-doc"></div>

## 📳 Mobile and Edge Device Deployment

<div id="fastdeploy-edge-models"></div>

<details open><summary><b> Mobile and Edge Model List（click to fold）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/115439700/212801271-5621419f-3997-4f00-94d5-63c8b6474aa8.png" height = "40"/>
</div>

| Task               | Model                                                                                        | Size(MB)   | Linux   | Android | Linux     | Linux                   | Linux                          | Linux                       | Linux                            | TBD ...  |
|:------------------:|:-----------------------------------------------------------------------------------------:|:--------:|:-------:|:-------:|:-------:|:-----------------------:|:------------------------------:|:---------------------------:|:--------------------------------:|:-------:|
| ---                | ---                                                                                       | ---      | ARM CPU | [ARM CPU](./java/android) | [Rockchip NPU<br>RK3588/RK3568/RK3566](./docs/en/build_and_install/rknpu2.md) | [Rockchip NPU<br>RV1109/RV1126/RK1808](./docs/en/build_and_install/rv1126.md) | [Amlogic NPU <br>A311D/S905D/C308X](./docs/en/build_and_install/a311d.md) | NXP NPU<br>i.MX&nbsp;8M&nbsp;Plus | TBD... |
| Classification     | [PaddleClas/ResNet50](examples/vision/classification/paddleclas)                         | 98        | ✅       | ✅       |  [✅](./examples/vision/classification/paddleclas/rknpu2)                             |      ✅                                |                                   |                                   |         |
| Classification     | [PaddleClas/PP-LCNet](examples/vision/classification/paddleclas)                         | 11.9      | ✅       | ✅       | ❔                             | ✅                                 | --                                | --                                | --      |
| Classification     | [PaddleClas/PP-LCNetv2](examples/vision/classification/paddleclas)                       | 26.6      | ✅       | ✅       | ❔                             | ✅                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/EfficientNet](examples/vision/classification/paddleclas)                     | 31.4      | ✅       | ✅       | ❔                             | ✅                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/GhostNet](examples/vision/classification/paddleclas)                         | 20.8      | ✅       | ✅       | ❔                             | ✅                                  | --                                | --                                | --      |
| Classification     | [PaddleClas/MobileNetV1](examples/vision/classification/paddleclas)                      | 17        | ✅       | ✅       | ❔                             | ✅                                  | --                                | --                                | --      |
| Classification     | [PaddleClas/MobileNetV2](examples/vision/classification/paddleclas)                      | 14.2      | ✅       | ✅       | ❔                             | ✅                                  | --                                | --                                | --      |
| Classification     | [PaddleClas/MobileNetV3](examples/vision/classification/paddleclas)                      | 22        | ✅       | ✅       | ❔                             | ✅                                    | ❔                                 | ❔                                 | --      |
| Classification     | [PaddleClas/ShuffleNetV2](examples/vision/classification/paddleclas)                     | 9.2       | ✅       | ✅       | ❔                             | ✅                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/SqueezeNetV1.1](examples/vision/classification/paddleclas)                   | 5         | ✅       | ✅       | ❔                             | ✅                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/Inceptionv3](examples/vision/classification/paddleclas)                      | 95.5      | ✅       | ✅       | ❔                             | ✅                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/PP-HGNet](examples/vision/classification/paddleclas)                         | 59        | ✅       | ✅       | ❔                             | ✅                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PicoDet_s](examples/vision/detection/paddledetection) | 4.9       | ✅       | ✅       | [✅](./examples/vision/detection/paddledetection/rknpu2)                             | ✅                                    | ✅                                 | ✅                                 | --      |
| Detection          | [YOLOv5](./examples/vision/detection/rkyolo) |        |  ❔     |  ❔      | [✅](./examples/vision/detection/rkyolo)                             | ❔                                   | ❔                                 | ❔                                 | --      |
| Face Detection     | [deepinsight/SCRFD](./examples/vision/facedet/scrfd)                                     | 2.5       | ✅       | ✅       | [✅](./examples/vision/facedet/scrfd/rknpu2)                             | --                                   | --                                | --                                | --      |
| Keypoint Detection | [PaddleDetection/PP-TinyPose](examples/vision/keypointdetection/tiny_pose)               | 5.5       | ✅       | ✅       | ❔                             | ❔                                    | ❔                                 | ❔                                 | --      |
| Segmentation       | [PaddleSeg/PP-LiteSeg(STDC1)](examples/vision/segmentation/paddleseg)                    | 32.2      | ✅       | ✅       | [✅](./examples/vision/segmentation/paddleseg/rknpu2)                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg-Lite](examples/vision/segmentation/paddleseg)                     | 0.556     | ✅       | ✅       | [✅](./examples/vision/segmentation/paddleseg/rknpu2)                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/HRNet-w18](examples/vision/segmentation/paddleseg)                            | 38.7      | ✅       | ✅       | [✅](./examples/vision/segmentation/paddleseg/rknpu2)                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg](examples/vision/segmentation/paddleseg)                          | 107.2     | ✅       | ✅       | [✅](./examples/vision/segmentation/paddleseg/rknpu2)                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/Unet](examples/vision/segmentation/paddleseg)                                 | 53.7      | ✅       | ✅       | [✅](./examples/vision/segmentation/paddleseg/rknpu2)                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/Deeplabv3](examples/vision/segmentation/paddleseg)                            | 150       | ❔       | ✅       | [✅](./examples/vision/segmentation/paddleseg/rknpu2)                             |                                      |                                   |                                   |         |
| OCR                | [PaddleOCR/PP-OCRv2](examples/vision/ocr/PP-OCRv2)                                       | 2.3+4.4   | ✅       | ✅       | ❔                             | --                                   | --                                | --                                | --      |
| OCR                | [PaddleOCR/PP-OCRv3](examples/vision/ocr/PP-OCRv3)                                       | 2.4+10.6  | ✅       | ❔       | ❔                             | ❔                                    | ❔                                 | ❔                                 | --      |

</div></details>

## ⚛️ Web and Mini Program Model List

<div id="fastdeploy-web-models"></div>

<details open><summary><b> Web and mini program model list(click to fold)</b></summary><div>

| Task               | Model                                                                                          | [web_demo](examples/application/js/web_demo) |
|:------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ---                | ---                                                                                         | [Paddle.js](examples/application/js)         |
| Detection          | [FaceDetection](examples/application/js/web_demo/src/pages/cv/detection)                    | ✅                                            |
| Detection          | [ScrewDetection](examples/application/js/web_demo/src/pages/cv/detection)                   | ✅                                            |
| Segmentation       | [PaddleSeg/HumanSeg](./examples/application/js/web_demo/src/pages/cv/segmentation/HumanSeg) | ✅                                            |
| Object Recognition | [GestureRecognition](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| Object Recognition | [ItemIdentification](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| OCR                | [PaddleOCR/PP-OCRv3](./examples/application/js/web_demo/src/pages/cv/ocr)                   | ✅                                            |

</div></details>

## 💐 Acknowledge

<div id="fastdeploy-acknowledge"></div>

We sincerely appreciate the open-sourced capabilities in [EasyEdge](https://ai.baidu.com/easyedge/app/openSource) as we adopt it for the SDK generation and download in this project.

## ©️ License

<div id="fastdeploy-license"></div>

FastDeploy is provided under the [Apache-2.0](./LICENSE).
