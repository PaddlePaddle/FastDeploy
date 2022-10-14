
# 预编译库安装

FastDeploy提供各平台预编译库，供开发者直接下载安装使用。当然FastDeploy编译也非常容易，开发者也可根据自身需求编译FastDeploy。

## GPU部署环境

### 环境要求
- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux(x64)/Windows 10(x64)

支持CPU和Nvidia GPU的部署，默认集成Paddle Inference、ONNX Runtime、OpenVINO以及TensorRT推理后端，Vision视觉模型模块，Text文本NLP模型模块

### Python安装

Release版本（当前最新0.2.1）安装
```
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html 
```

其中推荐使用Conda配置开发环境
```
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

### C++ SDK安装

Release版本（当前最新0.2.1）

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Linux x64 | [fastdeploy-linux-x64-gpu-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz) | g++ 8.2, CUDA 11.2, cuDNN 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-gpu-0.2.1.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip) | Visual Studio 16 2019, CUDA 11.2, cuDNN 8.2编译产出 |

## CPU部署环境

### 环境要求
- python >= 3.6
- OS: Linux(x64/aarch64)/Windows 10 x64/Mac OSX(x86/aarm64)

仅支持CPU部署，默认集成Paddle Inference、ONNX Runtime、OpenVINO, Vision视觉模型模块(Linux aarch64和Mac OSX下仅集成ONNX Runtime模块)， Text文本NLP模型模块。

### Python安装

Release版本（当前最新0.2.1）安装
```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## C++ SDK安装

Release版本（当前最新0.2.1）

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Linux x64 | [fastdeploy-linux-x64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.1.tgz) | g++ 8.2编译产出 |
| Windows x64 | [fastdeploy-win-x64-0.2.1.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-0.2.1.zip) | Visual Studio 16 2019编译产出 |
| Mac OSX x64 | [fastdeploy-osx-x86_64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-0.2.1.tgz) | - |
| Mac OSX arm64 | [fastdeploy-osx-arm64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-0.2.1.tgz) | - |
| Linux aarch64 | [fastdeploy-linux-aarch64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-0.2.0.tgz) | g++ 6.3.0编译产出 |

