[English](../../en/build_and_install/prebuilt.md) | 简体中文

# How to Install Prebuilt Library

FastDeploy provides pre-built libraries for developers to download and install directly. Meanwhile, FastDeploy also offers easy access to compile so that developers can compile FastDeploy according to their own needs.

## GPU Deployment Environment

### Environment Requirement

- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux(x64)/Windows 10(x64)

Supports CPU and Nvidia GPU deployment with default integration of Paddle Inference, ONNX Runtime, OpenVINO and TensorRT inference backends, Vision module, Text NLP module

### Python SDK

Install the released version（the newest 0.2.1 for now）

```
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html 
```

It is recommended to use Conda to configure the development environment

```
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

### C++ SDK

Install the released version（Latest 0.2.1）

| Platform    | File                                                                                                                  | Description                                               |
|:----------- |:--------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| Linux x64   | [fastdeploy-linux-x64-gpu-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz) | Compiled from g++ 8.2, CUDA 11.2, cuDNN 8.2               |
| Windows x64 | [fastdeploy-win-x64-gpu-0.2.1.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip)     | Compiled from Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2 |

## CPU Deployment Environment

### Environment Requirement

- python >= 3.6
- OS: Linux(x64/aarch64)/Windows 10 x64/Mac OSX(x86/aarm64)

Now it only supports CPU deployment with default integration of Paddle Inference, ONNX Runtime, OpenVINO, Vision module (only ONNX Runtime module is integrated under Linux aarch64 and Mac OSX), and Text NLP module.

### Python SDK

Install the released version（Latest 0.2.1 for now）

```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

### C++ SDK

Install the released version（Latest 0.2.1 for now）

| Platform      | File                                                                                                                  | Description                    |
|:------------- |:--------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| Linux x64     | [fastdeploy-linux-x64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.1.tgz)         | Compiled from g++ 8.2          |
| Windows x64   | [fastdeploy-win-x64-0.2.1.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-0.2.1.zip)             | Compiled from Visual Studio 16 |
| Mac OSX x64   | [fastdeploy-osx-x86_64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-0.2.1.tgz)       | -                              |
| Mac OSX arm64 | [fastdeploy-osx-arm64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-0.2.1.tgz)         | -                              |
| Linux aarch64 | [fastdeploy-linux-aarch64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-0.2.0.tgz) | Compiled from g++ 6.3.0        |
