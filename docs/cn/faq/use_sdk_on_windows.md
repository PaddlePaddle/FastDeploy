[English](../../en/faq/use_sdk_on_windows.md) | 中文

# 在 Windows 使用 FastDeploy C++ SDK

【**注意**】**编译只支持Release模型，不支持Debug模型**

## 1. 准备环境和Windows部署库
<div id="Environment"></div>  

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 8.0 (当WITH_GPU=ON)


1. 根据需求，选择下载对应的C++(CPU/GPU)部署库，下载文档见[安装文档说明](../build_and_install)
> 假定当前下载解压后的库路径在`D:\Download\fastdeploy-win-x64-gpu-x.x.x
2. 下载如下模型文件和测试图片
> https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz # (下载后解压缩)
> https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

## 2. 编译示例代码

本文档编译的示例代码可在解压的库中找到，编译工具依赖VS 2019的安装，**Windows打开x64 Native Tools Command Prompt for VS 2019命令工具**，通过如下命令开始编译

```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\detection\paddledetection\cpp

mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=D:\Download\fastdeploy-win-x64-gpu-x.x.x -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"

msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

如需使用Visual Studio 2019创建sln工程，或者CMake工程等方式编译，可参考如下文档
- [FastDeploy C++库在Windows上的多种使用方式](./use_sdk_on_windows_build.md)

## 3. 运行编译可执行程序

注意Windows上运行时，需要将FastDeploy依赖的库拷贝至可执行程序所在目录, 或者配置环境变量。FastDeploy提供了工具帮助我们快速将所有依赖库拷贝至可执行程序所在目录,通过如下命令将所有依赖的dll文件拷贝至可执行程序所在的目录
```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x

fastdeploy_init.bat install %cd% D:\Download\fastdeploy-win-x64-gpu-x.x.x\examples\vision\detection\paddledetection\cpp\build\Release
```

将dll拷贝到当前路径后，准备好模型和图片，使用如下命令运行可执行程序即可
```shell
cd Release
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0  # CPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 1  # GPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 2  # GPU + TensorRT
```

在此步骤中使用到的`fastdeploy_init.bat`提供更多其它功能，帮忙开发者使用，包括
- 查看SDK中所有dll, lib和include的路径
- 安装SDK中所有dll至指定目录
- 配置SDK环境变量

具体可参考如下文档
- [fastdeploy_init.bat工具的使用](./usage_of_fastdeploy_init_bat.md)
