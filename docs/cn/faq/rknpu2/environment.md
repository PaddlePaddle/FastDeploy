[English](../../../en/faq/rknpu2/environment.md) | 中文
# FastDeploy RKNPU2推理环境搭建

## 简介

在FastDeploy上部署模型前我们需要搭建一下开发环境。FastDeploy将环境搭建分成板端推理环境搭建和PC端模型转换环境搭建两个部分。

## 安装rknn_toolkit2

安装rknn_toolkit2中会存在依赖问题，这里介绍以下如何安装。


### 下载安装需要的软件包

安装rknntoolkit2之前，你需要安装以下依赖包。

```bash
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
libsm6 libgl1-mesa-glx libprotobuf-dev gcc g++
```

### 安装rknn_toolkit2环境

rknn_toolkit2依赖一些特定的包，因此建议使用conda新建一个虚拟环境进行安装。

```bash
# 创建虚拟环境
conda create -n rknn2 python=3.6
conda activate rknn2

# rknn_toolkit2对numpy存在特定依赖,因此需要先安装numpy==1.16.6
pip install numpy==1.16.6

# 安装rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
wget https://bj.bcebos.com/fastdeploy/third_libs/rknn_toolkit2-1.4.2b3+0bdd72ff-cp36-cp36m-linux_x86_64.whl
pip install rknn_toolkit2-1.4.2b3+0bdd72ff-cp36-cp36m-linux_x86_64.whl
```

## 资源链接

* [rknntoolkit2开发板下载地址 密码：rknn](https://eyun.baidu.com/s/3eTDMk6Y)
