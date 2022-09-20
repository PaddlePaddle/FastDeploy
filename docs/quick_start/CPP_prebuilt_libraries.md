# FastDeploy 预编译 C++ 库

FastDeploy提供了在Windows/Linux/Mac上的预先编译CPP部署库，开发者可以直接下载后使用，也可以自行编译代码，参考[编译文档](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/compile)

## 环境依赖

- cuda >= 11.2
- cudnn >= 8.0
- g++ >= 5.4(推荐8.2)

## 下载地址

### Linux x64平台

| 部署库下载地址 | 硬件 | 说明 |
| :------------- | :--- | :--- |
| [fastdeploy-linux-x64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.1.tgz) | CPU | g++ 8.2编译产出 |
| [fastdeploy-linux-x64-gpu-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz) | CPU/GPU | g++ 8.2, cuda 11.2, cudnn 8.2编译产出 |

### Windows 10 x64平台

| 部署库下载地址 | 硬件 | 说明 |
| :------------- | :--- | :--- |
| [fastdeploy-win-x64-0.2.1.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-0.2.1.zip) | CPU | Visual Studio 16 2019 编译产出 |
| [fastdeploy-win-x64-gpu-0.2.1.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip) | CPU/GPU | Visual Studio 16 2019，cuda 11.2, cudnn 8.2编译产出 |

### Linux aarch64平台

| 安装包 | 硬件 | 说明 |
| :----  | :-- | :--- |
| [fastdeploy-linux-aarch64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-0.2.0.tgz) | CPU | g++ 6.3.0编译产出 |
| [comming...] | Jetson | |

### Mac OSX平台

| 部署库下载地址 | 架构 |硬件 |
| :----  | :-- | :------ |
| [fastdeploy-osx-x86_64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-0.2.1.tgz) | x86 | CPU |
| [fastdeploy-osx-arm64-0.2.1.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-0.2.1.tgz) | arm64 | CPU |

## 其它文档

- [预编译Python安装包](./Python_prebuilt_wheels.md)
- [视觉模型C++/Python部署示例](../../examples/vision/)
