English | [简体中文](README_CN.md)

# FastDeploy Streamer

## Introduction

FastDeploy Streamer (FDStreamer) is an AI multimedia stream processing framework that arranges functions such as AI inference, audio and video decoding, encoding, and streaming in the form of pipeline, to enable end-to-end optimization and deployment of AI applications.

下图为FDStreamer架构图，绿色标注的部分已实现，未标注的部分仍在规划中。

- 用户可以利用极简的API和一个YAML配置文件来启动FDStreamer APP, 例如视频分析APP、视频解码APP
- FDStreamer APP层调用GStreamer API来构建Pipeline、处理回调函数等
- GStreamer的底层的element/plugin包括GStreamer框架内置的plugin（例如appsink、x265）、第三方SDK的plugin（例如DeepStream的nvinfer、nvtracker），以及FDStreamer提供的plugin（计划中）
- GStreamer plugin的底层会调用编解码硬件、CPU、GPU、NPU以及其他AI芯片等

The following figure is the architecture diagram of FDStreamer. The parts marked in green have been implemented, while the parts not marked are still under planning.

- Users can use a simple API and a YAML configuration file to start FDStreamer APP, such as video analytics APP, video decoder APP.
- FDStreamer APP calls GStreamer API to build Pipeline, handle callback functions, etc.
- The underlying elements/plugins of GStreamer include plugins built into the GStreamer framework (such as appsink, x265), plugins from 3rd SDKs (such as DeepStream’s nvinfer, nvtracker), and plugins provided by FDStreamer (under planning)
- GStreamer plugins run on different hardwares, such as codec hardware, CPU, GPU, NPU and other AI chips.

<img src='https://user-images.githubusercontent.com/15235574/208360353-49433f71-165d-43a8-bbc4-1461ab16544b.png' height="360px">

## Environment

### Jetson
- DeepStream 6.1+

### x86 GPU

Install DeepStream 6.1.1 and dependencies manually，or use below docker：
```
docker pull nvcr.io/nvidia/deepstream:6.1.1-devel
```

## Build

1. [Build FastDeploy](../docs/en/build_and_install), or download [FastDeploy prebuilt libraries](../docs/en/build_and_install/download_prebuilt_libraries.md)

2. Build Streamer
```
cd FastDeploy/streamer/
mkdir build && cd build/

# Download FastDeploy prebuilt libraries, please check `FastDeploy prebuilt libraries` above.
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j
```

3. Build and Run Example

| Example | Brief |
|:--|:--|
| [PP-YOLOE](./examples/ppyoloe) | Multiple input videos, PP-YOLOE object detection, NvTracker, Hardware codec, writing to mp4 file |
| [Video Decoder](./examples/video_decoder) | Video decoding using hardward |  
