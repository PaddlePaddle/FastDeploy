English | [中文](../../../cn/faq/rknpu2/environment.md) 
# FastDeploy RKNPU2 inference environment setup

## Introduction

We need to set up the development environment before deploying models on FastDeploy. The environment setup of FastDeploy is divided into two parts: the board-side inference environment setup and the PC-side model conversion environment setup.

## Board-side inference environment setup

Based on the feedback from developers, we provide two ways to set up the inference environment on the board: one-click script installation script and command line installation of development board dirver.

### Install via script

Most developers don't like complex command lines for installation, so FastDeploy provides a one-click way for developers to install stable RKNN. Refer to the following command to set up the board side environment

```bash
# Download and unzip rknpu2_device_install_1.4.0
wget https://bj.bcebos.com/fastdeploy/third_libs/rknpu2_device_install_1.4.0.zip
unzip rknpu2_device_install_1.4.0.zip

cd rknpu2_device_install_1.4.0
# RK3588 runs the following code 
sudo rknn_install_rk3588.sh
# RK356X  runs the following code 
sudo rknn_install_rk356X.sh
```

### Install via the command line 

For developers who want to try out the latest RK drivers, we provide a method to install them from scratch using the following command line. 

```bash
# Install the required packages 
sudo apt update -y
sudo apt install -y python3
sudo apt install -y python3-dev
sudo apt install -y python3-pip
sudo apt install -y gcc
sudo apt install -y python3-opencv
sudo apt install -y python3-numpy
sudo apt install -y cmake

# Download rknpu2
# RK3588 runs the following code 
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

# RK356X  runs the following code 
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
```

## Install rknn_toolkit2

There are dependency issues when installing the rknn_toolkit2. Here are the installation tutorial. 
rknn_toolkit2 depends on a few specific packages, so it is recommended to create a virtual environment using conda. The way to install conda is omitted and we mainly introduce how to install rknn_toolkit2.


### Download rknn_toolkit2
rknn_toolkit2 can usually be downloaded from git 
```bash
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
```

### Download and install the required packages 
```bash
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
libsm6 libgl1-mesa-glx libprotobuf-dev gcc g++
```

### Install rknn_toolkit2 environment 
```bash
# Create virtual environment
conda create -n rknn2 python=3.6
conda activate rknn2

# Install numpy==1.16.6 first because rknn_toolkit2 has a specific numpy dependency
pip install numpy==1.16.6

# Install rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
cd ~/Download /rknn-toolkit2-master/packages
pip install rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
```

## Resource links 

* [RKNPU2, rknntoolkit2 development board download  Password：rknn](https://eyun.baidu.com/s/3eTDMk6Y)

## Other documents 
- [RKNN model conversion document](./export.md)
