English | [中文](../../cn/build_and_install/download_prebuilt_libraries.md)

# How to Install Prebuilt Library

FastDeploy provides pre-built libraries for developers to download and install directly. Meanwhile, FastDeploy also offers easy access to compile so that developers can compile FastDeploy according to their own needs.

This document is divided into two parts:
- [1.GPU Deployment Environment](#1)
- [2.CPU Deployment Environment](#2)

<p id="1"></p>

## GPU Deployment Environment

### Environment Requirement

- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux(x64)/Windows 10(x64)

FastDeploy supports Computer Vision, Text and NLP model deployment on CPU and Nvidia GPU with Paddle Inference, ONNX Runtime, OpenVINO and TensorRT inference backends.

### Python SDK

Install the released version（the newest 1.0.3 for now）

```
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Install the Develop version（Nightly build）

```bash
pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

We recommend users to use Conda to configure the development environment.

```
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

### C++ SDK

Install the released version（Latest 1.0.3）

| Platform    | File                                                                                                                  | Description                                               |
|:----------- |:--------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| Linux x64 | [fastdeploy-linux-x64-gpu-1.0.3.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-1.0.3.tgz) | g++ 8.2, CUDA 11.2, cuDNN 8.2 |
| Windows x64 | [fastdeploy-win-x64-gpu-1.0.3.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-1.0.3.zip) | Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2 |

Install the Develop version（Nightly build）

| Platform    | File                                                                                                                  | Description                                               |
|:----------- |:--------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| Linux x64 | [fastdeploy-linux-x64-gpu-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-gpu-0.0.0.tgz) | g++ 8.2, CUDA 11.2, cuDNN 8.2 |
| Windows x64 | [fastdeploy-win-x64-gpu-0.0.0.zip](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-win-x64-gpu-0.0.0.zip) | Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2 |

<p id="2"></p>

## CPU Deployment Environment

### Environment Requirement

- python >= 3.6
- OS: Linux(x64/aarch64)/Windows 10 x64/Mac OSX(x86/aarm64)

FastDeploy supports computer vision, text and NLP model deployment on CPU with Paddle Inference, ONNX Runtime, OpenVINO inference backends. It should be noted that under Linux aarch64 and Mac OSX, only the ONNX Runtime is supported for now.

### Python SDK

Install the released version（Latest 1.0.3 for now）

```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Install the Develop version（Nightly build）

```bash
pip install fastdeploy-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

### C++ SDK

Install the released version（Latest 1.0.3 for now, Android is 1.0.3）

| Platform      | File                                                                                                                  | Description                    |
|:------------- |:--------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| Linux x64 | [fastdeploy-linux-x64-1.0.3.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-1.0.3.tgz) | g++ 8.2 |
| Windows x64 | [fastdeploy-win-x64-1.0.3.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-1.0.3.zip) | Visual Studio 16 2019 |
| Mac OSX x64 | [fastdeploy-osx-x86_64-1.0.3.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-1.0.3.tgz) | clang++ 10.0.0|
| Mac OSX arm64 | [fastdeploy-osx-arm64-1.0.3.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-1.0.3.tgz) | clang++ 13.0.0 |
| Linux aarch64 | [fastdeploy-osx-arm64-1.0.3.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-1.0.3.tgz) | gcc 6.3 |  
| Android armv7&v8 | [fastdeploy-android-1.0.3-shared.tgz](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-1.0.3-shared.tgz) | NDK 25, clang++, support arm64-v8a and armeabi-v7a  |      
| Android armv7&v8 | [fastdeploy-android-with-text-1.0.3-shared.tgz](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-with-text-1.0.3-shared.tgz) | contains Text API, such as FastTokenizer and UIE，NDK 25, clang++, support arm64-v8a and armeabi-v7a  |

## Java SDK

Install the released version（Android is 1.0.3 pre-release）

| Platform | File | Description |
| :--- | :--- | :---- |
| Android Java SDK | [fastdeploy-android-sdk-1.0.3.aar](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-sdk-1.0.3.aar) | NDK 20, minSdkVersion 15, targetSdkVersion 28 |  
| Android Java SDK | [fastdeploy-android-sdk-with-text-1.0.3.aar](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-sdk-with-text-1.0.3.aar) | contains Text API, such as FastTokenizer and UI, NDK 20, minSdkVersion 15, targetSdkVersion 28 |

Install the Develop version（Nightly build）

| Platform      | File                                                                                                                  | Description                    |
|:------------- |:--------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| Linux x64 | [fastdeploy-linux-x64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-0.0.0.tgz) | g++ 8.2 |
| Windows x64 | [fastdeploy-win-x64-0.0.0.zip](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-win-x64-0.0.0.zip) | Visual Studio 16 2019 |
| Mac OSX x64 | [fastdeploy-osx-arm64-0.0.0.tgz](https://bj.bcebos.com/fastdeploy/dev/cpp/fastdeploy-osx-arm64-0.0.0.tgz) | - |
| Mac OSX arm64 | [fastdeploy-osx-arm64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-osx-arm64-0.0.0.tgz) | clang++ 13.0.0 to compile |
| Linux aarch64 | - | - |  
| Android armv7&v8 | [fastdeploy-android-0.0.0-shared.tgz](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-0.0.0-shared.tgz) | NDK 25, clang++, support arm64-v8a and armeabi-v7a |  
| Android armv7&v8 | [fastdeploy-android-with-text-0.0.0-shared.tgz](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-with-text-0.0.0-shared.tgz) | contains Text API, such as FastTokenizer and UIE，NDK 25, clang++, support arm64-v8a and armeabi-v7a |  
| Android Java SDK | [fastdeploy-android-sdk-0.0.0.aar](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-sdk-0.0.0.aar) | NDK 20, minSdkVersion 15, targetSdkVersion 28 |  
| Android Java SDK | [fastdeploy-android-sdk-with-text-0.0.0.aar](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-sdk-with-text-0.0.0.aar) | contains Text API, such as FastTokenizer and UI, NDK 20, minSdkVersion 15, targetSdkVersion 28 |
