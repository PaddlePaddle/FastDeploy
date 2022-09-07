# FastDeploy C++ SDK

FastDeploy provides prebuilt CPP deployment libraries on Windows/Linux/Mac. Developers can download and use it directly or compile the code manually.

## Dependency

- cuda >= 11.2
- cudnn >= 8.0
- g++ >= 5.4 (8.2 is recommended)

## Download

### Linux x64

| SDK Download Link                                                                                                 | Hardware | Description                              |
|:--------------------------------------------------------------------------------------------------------------------- |:-------- |:---------------------------------------- |
| [fastdeploy-linux-x64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.0.tgz)         | CPU      | Built with g++ 8.2                       |
| [fastdeploy-linux-x64-gpu-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.0.tgz) | CPU/GPU  | Built with g++ 8.2, cuda 11.2, cudnn 8.2 |

### Windows 10 x64

| SDK Download Link                                                                                             | Hardware | Description                                           |
|:----------------------------------------------------------------------------------------------------------------- |:-------- |:----------------------------------------------------- |
| [fastdeploy-win-x64-0.2.0.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-0.2.0.zip)         | CPU      | Built with Visual Studio 16 2019                      |
| [fastdeploy-win-x64-gpu-0.2.0.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.0.zip) | CPU/GPU  | Built with Visual Studio 16 2019，cuda 11.2, cudnn 8.2 |

### Linux aarch64

| SDK Download Link                                                                                                   | Hardware | Description          |
|:--------------------------------------------------------------------------------------------------------------------- |:-------- |:-------------------- |
| [fastdeploy-linux-aarch64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-0.2.0.tgz) | CPU      | Built with g++ 6.3.0 |
| [comming...]                                                                                                          | Jetson   |                      |

### Mac OSX

| SDK Download Link                                                                                            | Architecture | Hardware |
|:--------------------------------------------------------------------------------------------------------------- |:------------ |:-------- |
| [fastdeploy-osx-x86_64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-x86_64-0.2.0.tgz) | x86          | CPU      |
| [fastdeploy-osx-arm64-0.2.0.tgz](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-osx-arm64-0.2.0.tgz)   | arm64        | CPU      |

## Other related docs

- [Install Python SDK](./download_python_sdk.md)
- [Example Vision Model deployment with C++/Python](../../examples/vision/)
