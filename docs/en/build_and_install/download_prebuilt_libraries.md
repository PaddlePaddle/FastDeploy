
# How to Install Prebuilt Library

FastDeploy provides pre-built libraries for developers to download and install directly. Meanwhile, FastDeploy also offers easy access to compile so that developers can compile FastDeploy according to their own needs.

## GPU Deployment Environment

### Environment Requirement

- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux(x64)/Windows 10(x64)

FastDeploy supports Computer Vision, Text and NLP model deployment on CPU and Nvidia GPU with Paddle Inference, ONNX Runtime, OpenVINO and TensorRT inference backends.

### Python SDK

Install the released version（the newest 0.4.0 for now）

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

Install the released version（Latest 0.4.0）

| Platform    | File                                                                                                                  | Description                                               |
|:----------- |:--------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| Linux x64   | [fastdeploy-linux-x64-gpu-0.4.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.4.0.tgz) | Compiled from g++ 8.2, CUDA 11.2, cuDNN 8.2               |
| Windows x64 | [fastdeploy-win-x64-gpu-0.4.0.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.4.0.zip)     | Compiled from Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2 |

Install the Develop version（Nightly build）

| Platform    | File                                                                                                                  | Description                                               |
|:----------- |:--------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| Linux x64   | [fastdeploy-linux-x64-gpu-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-gpu-0.0.0.tgz | Compiled from g++ 8.2, CUDA 11.2, cuDNN 8.2               |
| Windows x64 | [fastdeploy-win-x64-gpu-0.0.0.zip](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-win-x64-gpu-0.0.0.zip)     | Compiled from Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2 |

## CPU Deployment Environment

### Environment Requirement

- python >= 3.6
- OS: Linux(x64/aarch64)/Windows 10 x64/Mac OSX(x86/aarm64)

FastDeploy supports computer vision, text and NLP model deployment on CPU with Paddle Inference, ONNX Runtime, OpenVINO inference backends. It should be noted that under Linux aarch64 and Mac OSX, only the ONNX Runtime is supported for now.

### Python SDK

Install the released version（Latest 0.4.0 for now）

```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

Install the Develop version（Nightly build）

```bash
pip install fastdeploy-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```

### C++ SDK

Install the released version（Latest 0.4.0 for now）

| Platform      | File                                                                                                                  | Description                    |
|:------------- |:--------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| Linux x64     | [fastdeploy-linux-x64-0.4.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.4.0.tgz)         | Compiled from g++ 8.2          |
| Windows x64   | [fastdeploy-win-x64-0.4.0.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-0.4.0.zip)             | Compiled from Visual Studio 16 |
| Mac OSX x64   | [fastdeploy-osx-x86_64-0.4.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-0.4.0.tgz)       | -                              |
| Mac OSX arm64 | [fastdeploy-osx-arm64-0.4.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-0.4.0.tgz)         | -                              |
| Linux aarch64 | [fastdeploy-linux-aarch64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-0.2.0.tgz) | Compiled from g++ 6.3.0        |
| Android armv7&v8 | [fastdeploy-android-0.4.0-shared.tgz](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-0.4.0-shared.tgz) |  Compiled from NDK 25 and clang++, support arm64-v8a and armeabi-v7a |

Install the Develop version（Nightly build）

| Platform      | File                                                                                                                  | Description                    |
|:------------- |:--------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| Linux x64     | [fastdeploy-linux-x64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-0.0.0.tgz)         | Compiled from g++ 8.2          |
| Windows x64   | [fastdeploy-win-x64-0.0.0.zip](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-win-x64-0.0.0.zip)             | Compiled from Visual Studio 16 |
| Mac OSX x64   | -       | -                              |
| Mac OSX arm64 | [fastdeploy-osx-arm64-0.0.0.tgz](https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-osx-arm64-0.0.0.tgz)         | -                              |
| Linux aarch64 | - | -       |
| Android armv7&v8 | - | - |
