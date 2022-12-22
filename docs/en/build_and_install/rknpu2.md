English | [中文](../../cn/build_and_install/rknpu2.md)

# How to Build RKNPU2 Deployment Environment

## Notes
FastDeploy has initial support for RKNPU2 deployments. If you find bugs while using, please report an issue to give us feedback.

## Introduction
Currently, the following backend engines on the RK platform are supported:

| Backend                | Platform                   | Model format supported | Description                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | Compile switch is controlled by setting `ENABLE_ORT_BACKEND` ON or OFF(default)    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | Compile switch is controlled by setting `ENABLE_RKNPU2_BACKEND` ON or OFF(default) |


## How to Build and Install C++ SDK

RKNPU2 only supports compiling on linux, the following steps are done on linux.

### Update the driver and install the compiling environment


Before running the program, we need to install the latest RKNPU driver, which is currently updated to 1.4.0. To simplify the installation, here is a quick install script.

**Method 1: Install via script**
```bash
# Download and unzip rknpu2_device_install_1.4.0
wget https://bj.bcebos.com/fastdeploy/third_libs/rknpu2_device_install_1.4.0.zip
unzip rknpu2_device_install_1.4.0.zip

cd rknpu2_device_install_1.4.0
# For RK3588
sudo rknn_install_rk3588.sh
# For RK356X
sudo rknn_install_rk356X.sh
```

**Method 2: Install via gitee**
```bash
# Install necessary packages
sudo apt update -y
sudo apt install -y python3 
sudo apt install -y python3-dev 
sudo apt install -y python3-pip 
sudo apt install -y gcc
sudo apt install -y python3-opencv
sudo apt install -y python3-numpy
sudo apt install -y cmake

# download rknpu2
# For RK3588
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

# For RK356X
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
```

### Compile C++ SDK

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build

# Only a few key configurations are introduced here, see README.md for details.
# -DENABLE_ORT_BACKEND:     Whether to enable ONNX model, default OFF
# -DENABLE_RKNPU2_BACKEND:  Whether to enable RKNPU model, default OFF
# -RKNN2_TARGET_SOC:        Compile the SDK board model. Enter RK356X or RK3588 with case sensitive required.
cmake ..  -DENABLE_ORT_BACKEND=ON \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK3588 \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3
make -j8
make install
```

### Compile Python SDK

Python packages depend on `wheel`, please run `pip install wheel` before compiling.

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
cd python

export ENABLE_ORT_BACKEND=ON
export ENABLE_RKNPU2_BACKEND=ON
export ENABLE_VISION=ON
export RKNN2_TARGET_SOC=RK3588
python3 setup.py build
python3 setup.py bdist_wheel

cd dist

pip3 install fastdeploy_python-0.0.0-cp39-cp39-linux_aarch64.whl
```

## Model Deployment

Please refer to [RKNPU2 Model Deployment](../faq/rknpu2/rknpu2.md).