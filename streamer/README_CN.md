简体中文 | [English](README_EN.md)

# FastDeploy Streamer

## 简介

FastDeploy Streamer（FDStreamer）是一个多媒体流处理框架，以Pipeline的形式集成AI推理、音视频解码、编码、推流等功能，
赋能AI应用的端到端优化和部署。

目前FDStreamer只适配了NVIDIA GPU/Jetson平台，更多硬件和平台的支持敬请期待。

## 准备环境

### Jetson
- DeepStream 6.1+

### x86 GPU

手动安装DeepStream 6.1.1及其依赖项，或使用以下docker：
```
docker pull nvcr.io/nvidia/deepstream:6.1.1-devel
```
