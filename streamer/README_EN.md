English | [简体中文](README_CN.md)

# FastDeploy Streamer

## Introduction

FastDeploy Streamer (FDStreamer) is an AI multimedia stream processing framework that arranges functions such as AI inference, audio and video decoding, encoding, and streaming in the form of pipeline, to enable end-to-end optimization and deployment of AI applications.

Currently FDStreamer is only compatible with NVIDIA GPU/Jetson platform, please look forward to more hardware and platform support.

## Environment

### Jetson
- DeepStream 6.1+

### x86 GPU

Install DeepStream 6.1.1 and dependencies manually，or use below docker：
```
docker pull nvcr.io/nvidia/deepstream:6.1.1-devel
```

## Build

1. [Build FastDeploy](../../docs/en/build_and_install), or download [FastDeploy prebuilt libraries](../../docs/en/build_and_install/download_prebuilt_libraries.md)

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
