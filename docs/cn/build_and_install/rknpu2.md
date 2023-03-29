[English](../../en/build_and_install/rknpu2.md) | 简体中文

# FastDeploy RKNPU2 导航文档

RKNPU2指的是Rockchip推出的RK356X以及RK3588系列芯片的NPU。
目前FastDeploy已经初步支持使用RKNPU2来部署模型。
如果您在使用的过程中出现问题，请附带上您的运行环境，在Issues中反馈。

## FastDeploy RKNPU2 环境安装简介

如果您想在FastDeploy中使用RKNPU2推理引擎，你需要配置以下几个环境。

| 工具名          | 是否必须 | 安装设备  | 用途                              |  
|--------------|------|-------|---------------------------------|
| Paddle2ONNX  | 必装   | PC    | 用于转换PaddleInference模型到ONNX模型    |  
| RKNNToolkit2 | 必装   | PC    | 用于转换ONNX模型到rknn模型               |  
| RKNPU2       | 选装   | Board | RKNPU2的基础驱动，FastDeploy已经集成，可以不装 |

## 安装模型转换环境

模型转换环境需要在Ubuntu下完成，我们建议您使用conda作为python控制器，并使用python3.6作为您的模型转换环境。
例如您可以输入以下命令行完成对python3.6环境的创建

```bash
conda create -n rknn2 python=3.6
conda activate rknn2
```

### 安装必备的依赖软件包

在安装RKNNtoolkit2之前我们需要安装一下必备的软件包

```bash
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc g++
```


### 安装RKNNtoolkit2

目前，FastDeploy使用的转化工具版本号为1.4.2b3。如果你有使用最新版本的转换工具的需求，你可以在Rockchip提供的[百度网盘(提取码为rknn)](https://eyun.baidu.com/s/3eTDMk6Y)
中找到最新版本的模型转换工具。

```bash
# rknn_toolkit2对numpy存在特定依赖,因此需要先安装numpy==1.16.6
pip install numpy==1.16.6

# 安装rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
wget https://bj.bcebos.com/fastdeploy/third_libs/rknn_toolkit2-1.4.2b3+0bdd72ff-cp36-cp36m-linux_x86_64.whl
pip install rknn_toolkit2-1.4.2b3+0bdd72ff-cp36-cp36m-linux_x86_64.whl
```

## 安装FastDeploy C++ SDK

针对RK356X和RK3588的性能差异，我们提供了两种编译FastDeploy的方式。


### 板端编译FastDeploy C++ SDK

针对RK3588，其CPU性能较强，板端编译的速度还是可以接受的，我们推荐在板端上进行编译。以下教程在RK356X(debian10),RK3588(debian 11) 环境下完成。

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

mkdir build && cd build
cmake ..  -DENABLE_ORT_BACKEND=OFF \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK3588 \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.0
make -j8
make install
```

### 交叉编译FastDeploy C++ SDK

针对RK356X，其CPU性能较弱，我们推荐使用交叉编译进行编译。以下教程在Ubuntu 22.04环境下完成。

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

mkdir build && cd build
cmake ..  -DCMAKE_C_COMPILER=/home/zbc/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/home/zbc/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ \
          -DCMAKE_TOOLCHAIN_FILE=./../cmake/toolchain.cmake \
          -DTARGET_ABI=arm64 \
          -DENABLE_ORT_BACKEND=OFF \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK356X \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.0
make -j8
make install
```

如果你找不到编译工具，你可以复制[交叉编译工具](https://bj.bcebos.com/paddle2onnx/libs/gcc-linaro-6.3.1-2017.zip)进行下载。

### 配置环境变量

为了方便大家配置环境变量，FastDeploy提供了一键配置环境变量的脚本，在运行程序前，你需要执行以下命令

```bash
# 临时配置
source PathToFastDeploySDK/fastdeploy_init.sh

# 永久配置
source PathToFastDeploySDK/fastdeploy_init.sh
sudo cp PathToFastDeploySDK/fastdeploy_libs.conf /etc/ld.so.conf.d/
sudo ldconfig
```

## 编译FastDeploy Python SDK

除了NPU，Rockchip的芯片还有其他的一些功能。
这些功能大部分都是需要C/C++进行编程，因此如果您使用到了这些模块，我们不推荐您使用Python SDK.
Python SDK的编译暂时仅支持板端编译, 以下教程在RK3568(debian 10)、RK3588(debian 11) 环境下完成。Python打包依赖`wheel`，编译前请先执行`pip install wheel`


```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

cd python
export ENABLE_ORT_BACKEND=ON
export ENABLE_RKNPU2_BACKEND=ON
export ENABLE_VISION=ON

# 请根据你的开发版的不同，选择RK3588和RK356X
export RKNN2_TARGET_SOC=RK3588

# 如果你的核心板的运行内存大于等于8G，我们建议您执行以下命令进行编译。
python3 setup.py build
# 值得注意的是，如果你的核心板的运行内存小于8G，我们建议您执行以下命令进行编译。
python3 setup.py build -j1

python3 setup.py bdist_wheel
cd dist
pip3 install fastdeploy_python-0.0.0-cp39-cp39-linux_aarch64.whl
```

## 导航目录

* [RKNPU2开发环境搭建](../faq/rknpu2/environment.md)
* [编译FastDeploy](../faq/rknpu2/build.md)
* [RKNN模型导出建议](../faq/rknpu2/export.md)
* [RKNPU2模型速度一览表](../faq/rknpu2/rknpu2.md)
* [RKNPU2 常见问题合集](../faq/rknpu2/issues.md)
