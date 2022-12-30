# FastDeploy RKNPU2推理环境搭建

## 简介

在FastDeploy上部署模型前我们需要搭建一下开发环境。FastDeploy将环境搭建分成板端推理环境搭建和PC端模型转换环境搭建两个部分。

## 板端推理环境搭建

根据开发者的反馈，我们提供了一键安装脚本和命令行安装开发版驱动两种方式来安装板端的推理环境。

### 通过脚本安装

多数开发者不喜欢使用复杂的命令行来进行安装，FastDeploy贴心的为开发者提供了一键安装稳定版RKNN的方式。参考以下命令，即可一键安装板端编译环境

```bash
# 下载解压rknpu2_device_install_1.4.0
wget https://bj.bcebos.com/fastdeploy/third_libs/rknpu2_device_install_1.4.0.zip
unzip rknpu2_device_install_1.4.0.zip

cd rknpu2_device_install_1.4.0
# RK3588运行以下代码
sudo rknn_install_rk3588.sh
# RK356X运行以下代码
sudo rknn_install_rk356X.sh
```

### 通过命令行安装

在开发的过程中，有的开发者希望能够体验到最新的RK驱动，我们也提供了对应的安装方式，使用以下下命令行即可从零开始安装RKNN的驱动。

```bash
# 安装必备的包
sudo apt update -y
sudo apt install -y python3
sudo apt install -y python3-dev
sudo apt install -y python3-pip
sudo apt install -y gcc
sudo apt install -y python3-opencv
sudo apt install -y python3-numpy
sudo apt install -y cmake

# 下载rknpu2
# RK3588运行以下代码
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

# RK356X运行以下代码
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
```

## 安装rknn_toolkit2

安装rknn_toolkit2中会存在依赖问题，这里介绍以下如何安装。 rknn_toolkit2依赖一些特定的包，因此建议使用conda新建一个虚拟环境进行安装。
安装conda的方法百度有很多，这里跳过，直接介绍如何安装rknn_toolkit2。

### 下载rknn_toolkit2
一般可以通过git直接下载rknn_toolkit2
```bash
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
```

### 下载安装需要的软件包
```bash
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
libsm6 libgl1-mesa-glx libprotobuf-dev gcc g++
```

### 安装rknn_toolkit2环境
```bash
# 创建虚拟环境
conda create -n rknn2 python=3.6
conda activate rknn2

# rknn_toolkit2对numpy存在特定依赖,因此需要先安装numpy==1.16.6
pip install numpy==1.16.6

# 安装rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
cd ~/下载/rknn-toolkit2-master/packages
pip install rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
```

## 资源链接

* [RKNPU2、rknntoolkit2开发板下载地址 密码：rknn](https://eyun.baidu.com/s/3eTDMk6Y)

## 其他文档
- [RKNN 模型转换文档](./export.md)
