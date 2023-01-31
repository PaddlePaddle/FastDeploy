English | [简体中文](README_CN.md)

# FastDeploy Streamer

## Introduction

FastDeploy Streamer (FDStreamer) is an AI multimedia stream processing framework that arranges functions such as AI inference, audio and video decoding, encoding, and streaming in the form of pipeline, to enable end-to-end optimization and deployment of AI applications.

The following figure is the architecture diagram of FDStreamer. The parts marked in green have been implemented, while the parts not marked are still under planning.

- Users can use a simple API and a [YAML configuration file](docs/en/yaml_config.md) to start FDStreamer APP, such as video analytics APP, video decoder APP.
- FDStreamer APP calls GStreamer API to build Pipeline, handle callback functions, etc.
- The underlying elements/plugins of GStreamer include plugins built into the GStreamer framework (such as appsink, x265), plugins from 3rd SDKs (such as DeepStream’s nvinfer, nvtracker), and plugins provided by FDStreamer (under planning)
- GStreamer plugins run on different hardwares, such as codec hardware, CPU, GPU, NPU and other AI chips.

<p align="center">
<img src='https://user-images.githubusercontent.com/15235574/208366363-d1cb5b74-d4fe-431c-ab57-07f97c27731d.png' height="360px">
</p>

## Environment

### Jetson
- DeepStream 6.1+

### x86 GPU

Install DeepStream 6.1.1 and dependencies manually，or use below docker：
```
docker pull nvcr.io/nvidia/deepstream:6.1.1-devel
```

### CPU
- GSTreamer 1.14+

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

CMake Options:
- ENABLE_DEEPSTREAM, whether to enable NVIDIA DeepStream, ON by default.

3. Build and Run Example

| Example | Brief |
|:--|:--|
| [PP-YOLOE](./examples/ppyoloe) | Multiple input videos, PP-YOLOE object detection, NvTracker, Hardware codec, writing to mp4 file |
| [Video Decoder](./examples/video_decoder) | Video decoding using GPU or CPU |  
