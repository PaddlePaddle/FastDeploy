[English](../../en/build_and_install/download_prebuilt_libraries.md) | 简体中文

# 预编译库安装

FastDeploy提供各平台预编译库，供开发者直接下载安装使用。当然FastDeploy编译也非常容易，开发者也可根据自身需求编译FastDeploy。

本文分为两部分：
- [1.GPU部署环境](#1)
- [2.CPU部署环境](#2)

<p id="1"></p>

## GPU部署环境

### 环境要求
- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux(x64)/Windows 10(x64)

支持CPU和Nvidia GPU的部署，默认集成Paddle Inference、ONNX Runtime、OpenVINO以及TensorRT推理后端，Vision视觉模型模块，Text文本NLP模型模块

版本信息：Paddle Inference==2.4-dev5，ONNXRuntime==1.12.0，OpenVINO==2022.2.0.dev20220829，TensorRT==8.5.2.2

### Python安装

Release版本（当前最新1.0.7）安装
```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Develop版本（Nightly build）安装
```bash
pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

其中推荐使用Conda配置开发环境
```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

### C++ SDK安装

Release版本

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Linux x64 | [fastdeploy-linux-x64-gpu-1.0.7.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-1.0.7.tgz) | g++ 8.2, CUDA 11.2, cuDNN 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-gpu-1.0.7.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-1.0.7.zip) | Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2编译产出 |

Develop版本（Nightly build）

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Linux x64 | [fastdeploy-linux-x64-gpu-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-gpu-0.0.0.tgz) | g++ 8.2, CUDA 11.2, cuDNN 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-gpu-0.0.0.zip](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-win-x64-gpu-0.0.0.zip) | Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2编译产出 |

<p id="2"></p>

## CPU部署环境

### 环境要求
- python >= 3.6
- OS: Linux(x64/aarch64)/Windows 10 x64/Mac OSX(x86/aarm64)

仅支持CPU部署，默认集成Paddle Inference、ONNX Runtime、OpenVINO, Vision视觉模型模块(Linux aarch64和Mac OSX下仅集成ONNX Runtime模块)， Text文本NLP模型模块。

版本信息：Paddle Inference==2.4-dev5，ONNXRuntime==1.12.0，OpenVINO==2022.2.0.dev20220829

### Python安装

Release版本（当前最新1.0.7）安装
```bash
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Develop版本（Nightly build）安装
```bash
pip install fastdeploy-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

## C++ SDK安装

Release版本

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Linux x64 | [fastdeploy-linux-x64-1.0.7.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-1.0.7.tgz) | g++ 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-1.0.7.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-1.0.7.zip) | Visual Studio 16 2019编译产出 |
| Mac OSX x64 | [fastdeploy-osx-x86_64-1.0.7.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-1.0.7.tgz) | clang++ 10.0.0编译产出|
| Mac OSX arm64 | [fastdeploy-osx-arm64-1.0.7.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-1.0.7.tgz) | clang++ 13.0.0编译产出 |
| Linux aarch64 | [fastdeploy-linux-aarch64-1.0.7.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-1.0.7.tgz) | gcc 6.3编译产出 |  
| Android armv7&v8 | [fastdeploy-android-1.0.7-shared.tgz](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-1.0.7-shared.tgz) | CV API，NDK 25及clang++编译产出, 支持arm64-v8a及armeabi-v7a |
| Android armv7&v8 | [fastdeploy-android-with-text-1.0.7-shared.tgz](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-with-text-1.0.7-shared.tgz) | 包含 FastTokenizer、UIE 等 Text API，CV API，NDK 25 及 clang++编译产出, 支持arm64-v8a及armeabi-v7a |
| Android armv7&v8 | [fastdeploy-android-with-text-only-1.0.7-shared.tgz](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-with-text-only-1.0.7-shared.tgz) | 仅包含 FastTokenizer、UIE 等 Text API，NDK 25 及 clang++ 编译产出, 不包含 OpenCV 等 CV API。 支持 arm64-v8a 及 armeabi-v7a |

## Java SDK安装

Release版本（Java SDK 目前仅支持Android，版本为1.0.7）  

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Android Java SDK | [fastdeploy-android-sdk-1.0.7.aar](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-sdk-1.0.7.aar) | CV API，NDK 20 编译产出, minSdkVersion 15, targetSdkVersion 28 |
| Android Java SDK | [fastdeploy-android-sdk-with-text-1.0.7.aar](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-sdk-with-text-1.0.7.aar) | 包含 FastTokenizer、UIE 等 Text API，CV API，NDK 20 编译产出, minSdkVersion 15, targetSdkVersion 28 |


Develop版本（Nightly build）

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Linux x64 | [fastdeploy-linux-x64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-0.0.0.tgz) | g++ 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-0.0.0.zip](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-win-x64-0.0.0.zip) | Visual Studio 16 2019编译产出 |
| Mac OSX x64 | [fastdeploy-osx-x86_64-0.0.0.tgz](https://bj.bcebos.com/fastdeploy/dev/cpp/fastdeploy-osx-x86_64-0.0.0.tgz) | - |
| Mac OSX arm64 | [fastdeploy-osx-arm64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-osx-arm64-0.0.0.tgz) | clang++ 13.0.0编译产出 |
| Linux aarch64 | [fastdeploy-linux-aarch64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-aarch64-0.0.0.tgz) | - |  
| Android armv7&v8 | [fastdeploy-android-0.0.0-shared.tgz](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-0.0.0-shared.tgz) | CV API，NDK 25及clang++编译产出, 支持arm64-v8a及armeabi-v7a |
| Android armv7&v8 | [fastdeploy-android-with-text-0.0.0-shared.tgz](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-with-text-0.0.0-shared.tgz) | 包含 FastTokenizer、UIE 等 Text API，CV API，NDK 25及clang++编译产出, 支持arm64-v8a及armeabi-v7a |
| Android armv7&v8 | [fastdeploy-android-with-text-only-0.0.0-shared.tgz](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-with-text-only-0.0.0-shared.tgz) | 仅包含 FastTokenizer、UIE 等 Text API，NDK 25及clang++编译产出，不包含 OpenCV 等 CV API。 支持arm64-v8a及armeabi-v7a |
| Android Java SDK | [fastdeploy-android-sdk-0.0.0.aar](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-sdk-0.0.0.aar) | CV API，NDK 20 编译产出, minSdkVersion 15, targetSdkVersion 28 |
| Android Java SDK | [fastdeploy-android-sdk-with-text-0.0.0.aar](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-sdk-with-text-0.0.0.aar) | 包含 FastTokenizer、UIE 等 Text API，CV API，NDK 20 编译产出, minSdkVersion 15, targetSdkVersion 28 |
