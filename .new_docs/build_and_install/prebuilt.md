# 预编译库安装

FastDeploy提供各平台预编译库，供开发者直接下载安装使用。当然FastDeploy编译也非常容易，开发者也可根据自身需求编译FastDeploy。

## Python安装

### Nvidia GPU部署环境

#### 环境要求
- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux(x64)/Windows 10(x64)

支持CPU和Nvidia GPU的部署，默认集成Paddle Inference、ONNX Runtime、OpenVINO以及TensorRT推理后端，Vision视觉模型模块，Text文本NLP模型模块

Release版本（当前最新0.2.1）安装
```
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html 
```

其中推荐使用Conda配置开发环境
```
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

### CPU部署环境

- python >= 3.6
- OS: Linux(x64/aarch64)/Windows 10 x64/Mac OSX(x86/aarm64

