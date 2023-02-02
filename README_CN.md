[English](README_EN.md) | 简体中文 | [हिन्दी](./docs/docs_i18n/README_हिन्दी.md) | [日本語](./docs/docs_i18n/README_日本語.md) | [한국인](./docs/docs_i18n/README_한국인.md) | [Pу́сский язы́к](./docs/docs_i18n/README_Pу́сский_язы́к.md)

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
    <a href="/docs/cn/build_and_install"><b> 安装 </b></a>
    |
    <a href="docs/README_CN.md"><b> 使用文档 </b></a>
    |
    <a href="README_CN.md#fastdeploy-quick-start-python"><b> 快速开始 </b></a>
    |
    <a href="https://baidu-paddle.github.io/fastdeploy-api/"><b> API文档 </b></a>
    |
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><b> 更新日志 </b></a>
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


 **⚡️FastDeploy**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具， 支持**云边端**部署。提供超过 🔥160+ **Text**，**Vision**， **Speech**和**跨模态**模型📦**开箱即用**的部署体验，并实现🔚**端到端**的推理性能优化。包括 [物体检测](./examples/vision/detection)、[字符识别（OCR）](./examples/vision/ocr)、[人脸](./examples/vision/facedet)、[人像扣图](./examples/vision/matting)、[多目标跟踪系统](./examples/vision/tracking/pptracking)、[NLP](./examples/text)、[Stable Difussion文图生成](./examples/multimodal/stable_diffusion)、[TTS](./examples/audio/pp-tts) 等几十种任务场景，满足开发者**多场景、多硬件、多平台**的产业部署需求。

<div align="center">
    
<img src="https://user-images.githubusercontent.com/54695910/213087733-7f2ea97b-baa4-4b0d-9b71-202ff6032a30.png" >
    
</div>


## 🌠 近期更新

- ✨✨✨ **2023.01.17** 发布 [**YOLOv8**](./examples/vision/detection/paddledetection/) 在FastDeploy系列硬件的部署支持。 其中包括 [**Paddle YOLOv8**](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8) 以及 [**社区 ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics)
    - [**Paddle YOLOv8**](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8) 可以部署的硬件：[**Intel CPU**](./examples/vision/detection/paddledetection/python/infer_yolov8.py)、[**NVIDIA GPU**](./examples/vision/detection/paddledetection/python/infer_yolov8.py)、[**Jetson**](./examples/vision/detection/paddledetection/python/infer_yolov8.py)、[**飞腾**](./examples/vision/detection/paddledetection/python/infer_yolov8.py)、[**昆仑芯**](./examples/vision/detection/paddledetection/python/infer_yolov8.py)、[**昇腾**](./examples/vision/detection/paddledetection/python/infer_yolov8.py)、[**ARM CPU**](./examples/vision/detection/paddledetection/cpp/infer_yolov8.cc)、[**RK3588**](./examples/vision/detection/paddledetection/rknpu2/yolov8.md) 和 [**Sophgo TPU**](./examples/vision/detection/paddledetection/sophgo), 部分硬件包含 **Python** 部署和 **C++** 部署；
    -  [**社区 ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics) 可以部署的硬件：[**Intel CPU**](./examples/vision/detection/yolov8)、[**NVIDIA GPU**](./examples/vision/detection/yolov8)、[**Jetson**](./examples/vision/detection/yolov8)，均包含 **Python** 部署和 **C++** 部署；
    -  FastDeploy 一行模型API切换，可以实现**YOLOv8**、 **PP-YOLOE+**、**YOLOv5** 等模型性能对比。
 - 服务化部署结合VisualDL新增支持可视化部署。在FastDeploy容器中启动VDL服务后，即可在VDL界面修改模型配置、启动/管理模型服务、查看性能数据、发送请求等，详细操作可参考相关文档
    - [Serving可视化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/vdl_management.md)
    - [Serving可视化请求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/client.md#%E4%BD%BF%E7%94%A8fastdeploy-client%E8%BF%9B%E8%A1%8C%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AF%B7%E6%B1%82)
 
- **✨👥✨ 社区交流**

    - **Slack**：Join our [Slack community](https://join.slack.com/t/fastdeployworkspace/shared_invite/zt-1o50e4voz-zbiIneCNRf_eH99eS2NVLg) and chat with other community members about ideas

    - **微信**：扫描二维码，填写问卷加入技术社区，与社区开发者交流部署产业落地痛点问题

<div align="center">
    <img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg" width = "150" height = "150" />
</div>


<div id="fastdeploy-acknowledge"></div>

## 🌌 推理后端及能力

<font size=0.5em>

|  | <img src="https://user-images.githubusercontent.com/54695910/212475832-f32502e2-4be2-42fc-a380-2ae265417938.png" height = "26" />  | <img src="https://user-images.githubusercontent.com/54695910/212475828-240036b0-f06c-4c44-830a-d8b136099b09.png" height = "27"  /> |<img src="https://user-images.githubusercontent.com/54695910/212475827-b73a1191-b3a8-4ad5-b6f6-855b3d1ffc09.png" height = "26" />| <img src="https://user-images.githubusercontent.com/54695910/212475826-f52b0ef3-e512-49fe-9b52-e1b9d1e8b6c2.png" height = "30" />  | <img src="https://user-images.githubusercontent.com/54695910/212475825-9686ae78-bad9-4be9-852e-6ad23be209da.png" height = "30" />  | <img src="https://user-images.githubusercontent.com/54695910/212475822-067349d2-8c4a-4431-bf02-05387e2962a8.png" height = "30" />  |<img src="https://user-images.githubusercontent.com/54695910/212475820-5210efe0-3e9a-429a-ad9d-48e8da2ffd0b.png" height = "30" /> |
|:----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| X86_64&nbsp;CPU  |     |&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>&nbsp;&nbsp;&nbsp;    |   <img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>  | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>   |   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>   |
| NVDIA&nbsp;GPU    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    | <img src="https://user-images.githubusercontent.com/54695910/212474106-a297aa0d-9225-458e-b5b7-e31aec7cfa79.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>  |     |    |
|飞腾 CPU  |    |     |  <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>  | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473393-ae1958bd-ab7d-4863-94b9-32863e600ba1.svg" height = "17"/>   |    |    |   |
| 昆仑芯 XPU |    |    | <img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>    |     |   |    |
| 华为昇腾 NPU |     |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>| <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>    |   |     |    |
|Graphcore&nbsp;IPU   |    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    |    |  <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/>  |    |    |  |
| 算能    |     |     |     | <img src="https://user-images.githubusercontent.com/54695910/212473382-e3e9063f-c298-4b61-ad35-a114aa6e6555.svg" height = "17"/>   |    |  |    |
|Intel 显卡  |     |     |     | <img src="https://user-images.githubusercontent.com/54695910/212473392-9df374d4-5daa-4e2b-856b-6e50ff1e4282.svg" height = "17"/>   |    |   | |
|Jetson  | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    | <img src="https://user-images.githubusercontent.com/54695910/212545467-e64ee45d-bf12-492c-b263-b860cb1e172b.png" height = "25"/>    |<img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474106-a297aa0d-9225-458e-b5b7-e31aec7cfa79.svg" height = "17"/>   | <img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473391-92c9f289-a81a-4927-9f31-1ab3fa3c2971.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473556-d2ebb7cc-e72b-4b49-896b-83f95ae1fe63.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473190-fdf3cee2-5670-47b5-85e7-6853a8dd200a.svg" height = "17"/>  |     |    |
|ARM&nbsp;CPU |    |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212474104-d82f3545-04d4-4ddd-b240-ffac34d8a920.svg" height = "17"/>| <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/><br><img src="https://user-images.githubusercontent.com/54695910/212473393-ae1958bd-ab7d-4863-94b9-32863e600ba1.svg" height = "17"/>   |    | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>  |  <img src="https://user-images.githubusercontent.com/54695910/212473393-ae1958bd-ab7d-4863-94b9-32863e600ba1.svg" height = "17"/>  |
|RK3588等 |   |    | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473387-2559cc2a-024b-4452-806c-6105d8eb2339.svg" height = "17"/>  |    |    |    |
|RV1126等 |    |    | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>    |     |     |    |
| 晶晨 |   |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    | <img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>   |     |     |   |
|  恩智浦 |   |     | <img src="https://user-images.githubusercontent.com/54695910/212474105-38051192-9a1c-4b24-8ad1-f842fb0bf39d.svg" height = "17"/>    |<img src="https://user-images.githubusercontent.com/54695910/212473389-8c341bbe-30d4-4a28-b50a-074be4e98ce6.svg" height = "17"/>   |     |    |    |

</font>


## 🔮 文档教程
- [✴️ Python SDK快速开始](#fastdeploy-quick-start-python)  
- [✴️ C++ SDK快速开始](#fastdeploy-quick-start-cpp)
- **安装文档**
    - [预编译库下载安装](docs/cn/build_and_install/download_prebuilt_libraries.md)
    - [GPU部署环境编译安装](docs/cn/build_and_install/gpu.md)
    - [CPU部署环境编译安装](docs/cn/build_and_install/cpu.md)
    - [IPU部署环境编译安装](docs/cn/build_and_install/ipu.md)
    - [昆仑芯XPU部署环境编译安装](docs/cn/build_and_install/kunlunxin.md)
    - [瑞芯微RV1126部署环境编译安装](docs/cn/build_and_install/rv1126.md)
    - [瑞芯微RK3588部署环境编译安装](docs/cn/build_and_install/rknpu2.md)
    - [晶晨A311D部署环境编译安装](docs/cn/build_and_install/a311d.md)
    - [华为昇腾部署环境编译安装](docs/cn/build_and_install/huawei_ascend.md)
    - [Jetson部署环境编译安装](docs/cn/build_and_install/jetson.md)
    - [Android平台部署环境编译安装](docs/cn/build_and_install/android.md)
- **快速使用**
    - [PP-YOLOE Python部署示例](docs/cn/quick_start/models/python.md)
    - [PP-YOLOE C++部署示例](docs/cn/quick_start/models/cpp.md)
- **不同后端使用**
    - [Runtime Python使用示例](docs/cn/quick_start/runtime/python.md)
    - [Runtime C++使用示例](docs/cn/quick_start/runtime/cpp.md)
    - [如何配置模型部署的推理后端](docs/cn/faq/how_to_change_backend.md)
- **服务化部署**
    - [服务化部署镜像编译安装](serving/docs/zh_CN/compile.md)
    - [服务化部署](serving)
- **API文档**
    - [Python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/)
    - [C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/)
    - [Android Java API文档](java/android)
- **性能调优**
    - [量化加速](docs/cn/quantize.md)
    - [多线程多进程使用](/tutorials/multi_thread)
- **常见问题**
    - [1. Windows上C++ SDK如何使用](docs/cn/faq/use_sdk_on_windows.md)
    - [2. Android上如何使用FastDeploy C++ SDK](docs/cn/faq/use_cpp_sdk_on_android.md)
    - [3. TensorRT使用中的一些技巧](docs/cn/faq/tensorrt_tricks.md)
- **更多FastDeploy部署模块**
    - [Benchmark测试](benchmark)
- **模型支持列表**
    - [🖥️ 服务端 模型支持列表](#fastdeploy-server-models)
    - [📳 移动端和端侧 模型支持列表](#fastdeploy-edge-models)
    - [⚛️ Web和小程序 模型支持列表](#fastdeploy-web-models)
- **💕开发者贡献**
    - [增加新模型](docs/cn/faq/develop_a_new_model.md)



<div id="fastdeploy-quick-start-python"></div>

## 快速开始💨

<details Open>

<summary><b>Python SDK快速开始（点开收缩）</b></summary><div>

### 🎆 快速安装

#### 🔸 前置依赖

- CUDA >= 11.2、cuDNN >= 8.0、Python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

#### 🔸 安装GPU版本

```bash
pip install numpy opencv-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### [🔸 Conda安装(推荐✨)](docs/cn/build_and_install/download_prebuilt_libraries.md)

```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

#### 🔸 安装CPU版本

```bash
pip install numpy opencv-python fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

### 🎇 Python 推理示例

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
result = model.predict(im)
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)

```

</div></details>

<div id="fastdeploy-quick-start-cpp"></div>

<details close>

<summary><b>C++ SDK快速开始（点开查看详情）</b></summary><div>


### 🎆 安装

- 参考[C++预编译库下载](docs/cn/build_and_install/download_prebuilt_libraries.md)文档  

#### 🎇 C++ 推理示例

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
  model.Predict(im, &res);

  auto vis_im = vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
}
```

</div></details>

更多部署案例请参考[模型部署示例](examples) .


<div id="fastdeploy-server-models"></div>


## ✴️ ✴️ 服务端模型支持列表 ✴️ ✴️ 

符号说明: (1)  ✅ : 已经支持; (2) ❔: 正在进行中; (3) N/A : 暂不支持. <br>

<details open><summary><b> 服务端模型支持列表（点击可收缩）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/115439700/212800663-894e9a7a-6d68-4b0b-bcd2-045732d08887.png" height ="40"/>
</div>

| 任务场景                   | 模型                                                                                           | Linux                                            | Linux      | Win     | Win        | Mac     | Mac     | Linux       | Linux           | Linux         | Linux         | Linux   | Linux   | Linux   |
|:----------------------:|:--------------------------------------------------------------------------------------------:|:------------------------------------------------:|:----------:|:-------:|:----------:|:-------:|:-------:|:-----------:|:---------------:|:-------------:|:-------------:|:-------:|:-------:|:-------:|
| ---                    | ---                                                                                          | X86 CPU                                          | NVIDIA GPU | X86 CPU | NVIDIA GPU | X86 CPU | Arm CPU | AArch64 CPU | 飞腾D2000 aarch64 | [NVIDIA Jetson](./docs/cn/build_and_install/jetson.md) | [Graphcore IPU](./docs/cn/build_and_install/ipu.md) | [昆仑芯 XPU](./docs/cn/build_and_install/kunlunxin.md) |[华为 昇腾](./docs/cn/build_and_install/huawei_ascend.md) |  [Serving](./serving) |
| Classification         | [PaddleClas/ResNet50](./examples/vision/classification/paddleclas)                           | [✅](./examples/vision/classification/paddleclas) | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ✅                           | ✅       | ✅       |✅       |
| Classification         | [TorchVison/ResNet](examples/vision/classification/resnet)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       |✅       | ❔       |
| Classification         | [ultralytics/YOLOv5Cls](examples/vision/classification/yolov5cls)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                | ✅                           | ❔                           | ✅       | ❔       |❔       |
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

<div id="fastdeploy-edge-models"></div>

## 📳 移动端和端侧 模型支持列表

<details open><summary><b> 端侧模型支持列表（点击可收缩）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/115439700/212800663-894e9a7a-6d68-4b0b-bcd2-045732d08887.png" height ="40"/>
</div>

| 任务场景               | 模型                                                                                        | 大小(MB)   | Linux   | Android | Linux     | Linux                   | Linux                          | Linux                       | Linux                            | 更新中...  |
|:------------------:|:-----------------------------------------------------------------------------------------:|:--------:|:-------:|:-------:|:-------:|:-----------------------:|:------------------------------:|:---------------------------:|:--------------------------------:|:-------:|
| ---                | ---                                                                                       | ---      | ARM CPU | [ARM CPU](./java/android) | [瑞芯微NPU<br>RK3588/RK3568/RK3566](./docs/cn/build_and_install/rknpu2.md) | [瑞芯微NPU<br>RV1109/RV1126/RK1808](./docs/cn/build_and_install/rv1126.md) | [晶晨NPU <br>A311D/S905D/C308X](./docs/cn/build_and_install/a311d.md) | 恩智浦NPU<br>i.MX&nbsp;8M&nbsp;Plus | 更新中... |
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

## ⚛️ Web和小程序 模型支持列表

<div id="fastdeploy-web-models"></div>

<details open><summary><b> Web和小程序部署支持列表（点击可收缩）</b></summary><div>

| 任务场景               | 模型                                                                                          | [web_demo](examples/application/js/web_demo) |
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

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，在此表示感谢。

## ©️ License

<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
