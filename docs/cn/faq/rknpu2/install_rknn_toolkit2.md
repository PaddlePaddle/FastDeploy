# 安装rknn_toolkit2仓库

## 下载rknn_toolkit2

rknn_toolkit2的下载一般有两种方式，以下将一一介绍:

* github仓库下载

  github仓库中提供了稳定版本的rknn_toolkit2下载
  ```bash
  git clone https://github.com/rockchip-linux/rknn-toolkit2.git
  ```
  
* 百度网盘下载

  在有些时候，如果稳定版本的rknn_toolkit2存在bug，不满足模型部署的要求，我们也可以使用百度网盘下载beta版本的rknn_toolkit2使用。其安装方式与
  稳定版本一致
  ```text
  链接：https://eyun.baidu.com/s/3eTDMk6Y 密码：rknn
  ```
  
## 安装rknn_toolkit2

安装rknn_toolkit2中会存在依赖问题，这里介绍以下如何安装。首先，因为rknn_toolkit2依赖一些特定的包，因此建议使用conda新建一个虚拟环境进行安装。
安装conda的方法百度有很多，这里跳过，直接介绍如何安装rknn_toolkit2。


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

## 其他文档
- [onnx转换rknn文档](./export.md)