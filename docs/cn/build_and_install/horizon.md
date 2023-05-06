[English](../../en/build_and_install/horizon.md) | 简体中文

# FastDeploy Horizon 导航文档

Horizon指的是地平线推出的旭日X3系列芯片的BPU。
目前FastDeploy已经初步支持使用Horizon来部署模型。
如果您在使用的过程中出现问题，请附带上您的运行环境，在Issues中反馈。

## FastDeploy Horizon 环境安装简介

如果您想在FastDeploy中使用Horizon推理引擎，你需要配置以下几个环境。

| 工具名          | 是否必须 | 安装设备  | 用途                              |  
|--------------|------|-------|---------------------------------|
| Paddle2ONNX  | 必装   | PC    | 用于转换PaddleInference模型到ONNX模型    |  
| 地平线XJ3芯片工具链镜像 | 必装   | PC    | 用于转换ONNX模型到地平线模型               |  
| 地平线 XJ3 OpenExplorer       | 必装   | PC | 地平线模型转换的关键头文件和动态库 |

## 安装模型转换环境

地平线提供了一套完整的模型转换环境（XJ3芯片工具链镜像），FastDeploy采用的镜像版本为[2.5.2](ftp://vrftp.horizon.ai/Open_Explorer_gcc_9.3.0/2.5.2/docker_openexplorer_ubuntu_20_xj3_gpu_v2.5.2_py38.tar.gz)，你可以通过地平线开发者平台获取。


## 安装必备的依赖软件包

地平线同样提供了一整套工具包(地平线 XJ3 OpenExplorer)，FastDeploy采用的开发包版本为[2.5.2](ftp://vrftp.horizon.ai/Open_Explorer_gcc_9.3.0/2.5.2/horizon_xj3_openexplorer_v2.5.2_py38_20230331.tar.gz),你可以通过地平线开发者平台获取。

由于板端CPU性能较弱，所以推荐在PC机上进行交叉编译。以下教程在地平线提供的docker环境下完成。

### 启动docker环境
将地平线XJ3芯片工具链镜像下载到本地之后，执行如下命令，将镜像包导入docker环境：

```bash
docker load < docker_openexplorer_ubuntu_20_xj3_gpu_v2.5.2_py38.tar.gz
```
将依赖的软件包下载至本地之后，解压：
```bash
tar -xvf horizon_xj3_openexplorer_v2.5.2_py38_20230331.tar.gz
```
解压完成之后，cd至改目录：
```bash
cd horizon_xj3_open_explorer_v2.5.2-py38_20230331/
```

根目录下有运行docker的脚本，运行以下命令：
```bash
sh run_docker.sh /home gpu
```

第一个目录为要挂载到容器上的目录，后一个参数为该docker启用gpu进行加速。

至此，所需环境准备完毕。

## 安装FastDeploy C++ SDK
下载交叉编译工具，[gcc_linaro_6.5.0_2018.12_x86_64_aarch64_linux_gnu](https://bj.bcebos.com/fastdeploy/third_libs/gcc_linaro_6.5.0_2018.12_x86_64_aarch64_linux_gnu.tar.xz)，建议解压后放到`/opt`目录下。
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

mkdir build && cd build
cmake ..  -DCMAKE_C_COMPILER=/opt/gcc_linaro_6.5.0_2018.12_x86_64_aarch64_linux_gnu/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/opt/gcc_linaro_6.5.0_2018.12_x86_64_aarch64_linux_gnu/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ \
          -DCMAKE_TOOLCHAIN_FILE=./../cmake/toolchain.cmake \
          -DTARGET_ABI=arm64 \
          -WITH_TIMVX=ON \
          -DENABLE_HORIZON_BACKEND=ON \
          -DENABLE_VISION=ON \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.0 \
          -Wno-dev ..
make -j16
make install
```

