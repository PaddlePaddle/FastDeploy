# How to build Huawei Ascend Deployment Environment

Based on the Paddle-Lite backend, FastDeploy supports model inference on Huawei's Ascend NPU.
For more detailed information, please refer to: [PaddleLite Deployment Example](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/huawei_ascend_npu.md).

This document describes how to compile PaddleLite-based C++ and Python FastDeploy source code under ARM Linux OS environment to generate prediction libraries for Huawei Sunrise NPU as the target hardware.

For more compilation options, please refer to the [FastDeploy compilation options description](./README.md)

##  Huawei Ascend Environment Preparation
- Atlas 300I Pro, see detailes at [Spec Sheet](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300i-pro)
- Install the driver and firmware package (Driver and Firmware) for the Atlas 300I Pro
- Download the matching driver and firmware package at:
  - https://www.hiascend.com/hardware/firmware-drivers?tag=community（Community Edition）
  - https://www.hiascend.com/hardware/firmware-drivers?tag=commercial（Commercial version）
  - driver：Atlas-300i-pro-npu-driver_5.1.rc2_linux-aarch64.run
  - firmware：Atlas-300i-pro-npu-firmware_5.1.rc2.run
- Installing drivers and firmware packages:

```shell
$ chmod +x *.run

$ ./Atlas-300i-pro-npu-driver_5.1.rc2_linux-aarch64.run --full
$ ./Atlas-300i-pro-npu-firmware_5.1.rc2.run --full

$ reboot
# Check the driver information to confirm successful installation
$ npu-smi info
```
- More system and detailed information is available in the [Ascend Hardware Product Documentation](https://www.hiascend.com/document?tag=hardware)

## Compilation environment construction

### Host environment requirements  
- os: ARM-Linux
- gcc, g++, git, make, wget, python, pip, python-dev, patchelf
- cmake (version 3.10 or above recommended)

### Using Docker development environment
In order to ensure consistency with the FastDeploy verified build environment, it is recommended to use the Docker development environment for configuration.

```shell
# Download Dockerfile
$ wget https://bj.bcebos.com/fastdeploy/test/Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile
# Create docker images
$ docker build --network=host -f Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile -t paddlelite/ascend_aarch64:cann_5.1.rc2 .
# Create container
$ docker run -itd --privileged --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_aarch64:cann_5.1.rc2 /bin/bash
# Enter the container
$ docker exec -it ascend-aarch64 /bin/bash
# Verify that the Ascend environment for the container is created successfully
$ npu-smi info
```
Once the above steps are successful, the user can start compiling FastDeploy directly from within docker.

Note:
- If you want to use another CANN version in Docker, please update the CANN download path in the Dockerfile file, and update the corresponding driver and firmware. The current default in Dockerfile is [CANN 5.1.RC2](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%205.1.RC2/Ascend-cann-toolkit_5.1.RC2_linux-aarch64.run).
- If users do not want to use docker, you can refer to [Compile Environment Preparation for ARM Linux Environments](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/arm_linux_compile_arm_linux.rst) provided by PaddleLite and configure your own compilation environment, and then download and install the proper CANN packages to complete the configuration.

## C++ FastDeploy library compilation based on PaddleLite
After setting up the compilation environment, the compilation command is as follows.

```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# CMake configuration with Ascend
cmake -DWITH_ASCEND=ON  \
      -DCMAKE_INSTALL_PREFIX=fastdeploy-ascend \
      -DENABLE_VISION=ON \
      ..

# Build FastDeploy Ascend C++ SDK
make -j8
make install
```  
When the compilation is complete, the fastdeploy-ascend directory is created in the current build directory, indicating that the PadddleLite-based FastDeploy library has been compiled.

## Compiling Python FastDeploy Libraries Based on PaddleLite

```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_ASCEND_PYTHON=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel

#After the compilation is complete, please install the whl package in the dist folder of the current directory.
```

Deploying PaddleClas Classification Model on Huawei Ascend NPU using C++ please refer to: [PaddleClas Huawei Ascend NPU C++ Deployment Example](../../../examples/vision/classification/paddleclas/ascend/cpp/README.md)

Deploying PaddleClas classification model on Huawei Ascend NPU using Python please refer to: [PaddleClas Huawei Ascend NPU Python Deployment Example](../../../examples/vision/classification/paddleclas/ascend/python/README.md)
