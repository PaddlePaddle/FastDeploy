简体中文 | [English](README_EN.md)

# FastDeploy Streamer

## 简介

FastDeploy Streamer（FDStreamer）是一个AI多媒体流处理框架，以Pipeline的形式编排AI推理、音视频解码、编码、推流等功能，
赋能AI应用的端到端优化和部署。

下图为FDStreamer架构图，绿色标注的部分已实现，未标注的部分仍在规划中。

- 用户可以利用极简的API和一个[YAML配置文件](docs/cn/yaml_config.md)来启动FDStreamer APP, 例如视频分析APP、视频硬解码APP
- FDStreamer APP层调用GStreamer API来构建Pipeline、处理回调函数等
- GStreamer的底层的element/plugin包括GStreamer框架内置的plugin（例如appsink、x265）、第三方SDK的plugin（例如DeepStream的nvinfer、nvtracker），以及FDStreamer提供的plugin（计划中）
- GStreamer plugin的底层会调用编解码硬件、CPU、GPU、NPU以及其他AI芯片等

<p align="center">
<img src='https://user-images.githubusercontent.com/15235574/208366363-d1cb5b74-d4fe-431c-ab57-07f97c27731d.png' height="360px">
</p>

## 准备环境

### Jetson
- DeepStream 6.1+

### x86 GPU

手动安装DeepStream 6.1.1及其依赖项，或使用以下docker：
```
docker pull nvcr.io/nvidia/deepstream:6.1.1-devel
```

### CPU
- GSTreamer 1.14+

## 编译和运行

1. [编译FastDeploy](../docs/cn/build_and_install), 或直接下载[FastDeploy预编译库](../docs/cn/build_and_install/download_prebuilt_libraries.md)

2. 编译Streamer
```
cd FastDeploy/streamer/
mkdir build && cd build/

# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j
```

编译选项：
- ENABLE_DEEPSTREAM，是否使用NVIDIA DeepStream，非NVIDIA GPU的环境需关闭此选项，默认ON

3. 编译和运行Example

| Example | 简介 |
|:--|:--|
| [PP-YOLOE](./examples/ppyoloe) | 多路视频接入，PP-YOLOE目标检测，NVTracker跟踪，硬编解码，写入mp4文件 |
| [Video Decoder](./examples/video_decoder) | 视频硬解码和软解码 |  
