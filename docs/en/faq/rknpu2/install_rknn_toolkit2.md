English | [中文](../../cn/faq/rknpu2/install_rknn_toolkit2.md) 
# RKNN-Toolkit2 Installation

## Download

Here are two methods to download RKNN-Toolkit2:

* Download from github library

A stable version of RKNN-Toolkit2 is available on github.
  ```bash
  git clone https://github.com/rockchip-linux/rknn-toolkit2.git
  ```
  
* Download from Baidu Netdisk

    In some cases, if the stable version has bugs and does not meet the requirements for model deployment, you can also use the beta version by downloading it from Baidu Netdisk. The installation way is the same as its stable version.
  ```text
  link：https://eyun.baidu.com/s/3eTDMk6Y password：rknn
  ```
  
## Installation

There will be dependency issues during the installation. Since some specific packages are required, it is recommended that you create a new conda environment at first.
You may get conda installation instruction on google, let's just skip it and introduce how to install RKNN-Toolkit2.


### Download and Install the packages required
```bash
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
libsm6 libgl1-mesa-glx libprotobuf-dev gcc g++
```

### Environment for installing RKNN-Toolkit2
```bash
# Create a new environment
conda create -n rknn2 python=3.6
conda activate rknn2

# RKNN-Toolkit2 has a specific dependency on numpy
pip install numpy==1.16.6

# Install rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl 
cd ~/download/rknn-toolkit2-master/packages
pip install rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl 
```

## Other Documents
- [How to convert ONNX to RKNN](./export.md)
