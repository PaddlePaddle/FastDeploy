# How to build Huawei Ascend Deployment Environment

Based on the Paddle-Lite backend, FastDeploy supports model inference on Huawei's Ascend NPU.
For more detailed information, please refer to: [Paddle Lite Deployment Example](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/huawei_ascend_npu.md).

This document describes how to compile C++ and Python FastDeploy source code under ARM/X86_64 Linux OS environment to generate prediction libraries for Huawei Sunrise NPU as the target hardware.

For more compilation options, please refer to the [FastDeploy compilation options description](./README.md)

##  Huawei Ascend Environment Preparation
- Atlas 300I Pro, see detailes at [Spec Sheet](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300i-pro)
- Install the driver and firmware package (Driver and Firmware) for the Atlas 300I Pro
- Download the matching driver and firmware package at:
  - https://www.hiascend.com/hardware/firmware-drivers?tag=community（Community Edition）
  - https://www.hiascend.com/hardware/firmware-drivers?tag=commercial（Commercial version）
  - driver：Atlas-300i-pro-npu-driver_5.1.rc2_linux-aarch64.run (aarch64 as example)
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
- os: ARM-Linux, X86_64-Linux
- gcc, g++, git, make, wget, python, pip, python-dev, patchelf
- cmake (version 3.10 or above recommended)

### Using Docker development environment
In order to ensure consistency with the FastDeploy verified build environment, it is recommended to use the Docker development environment for configuration.

On aarch64 platform,
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
On X86_64 platform,
```shell
# Download Dockerfile
$ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/Ascend_ubuntu18.04_x86_5.1.rc1.alpha001.Dockerfile
# Create docker images
$ docker build --network=host -f Ascend_ubuntu18.04_x86_5.1.rc1.alpha001.Dockerfile -t paddlelite/ascend_x86:cann_5.1.rc1.alpha001 .
# Create container
$ docker run -itd --privileged --name=ascend-x86 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_x86:cann_5.1.1.alpha001 /bin/bash
# Enter the container
$ docker exec -it ascend-x86 /bin/bash
# Verify that the Ascend environment for the container is created successfully
$ npu-smi info
```

Once the above steps are successful, the user can start compiling FastDeploy directly from within docker.

Note:
- If you want to use another CANN version in Docker, please update the CANN download path in the Dockerfile file, and update the corresponding driver and firmware. The current default in aarch64 Dockerfile is [CANN 5.1.RC2](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%205.1.RC2/Ascend-cann-toolkit_5.1.RC2_linux-aarch64.run), in x86_64 is [CANN 5.1.RC1](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/5.1.RC1.alpha001/Ascend-cann-toolkit_5.1.RC1.alpha001_linux-x86_64.run).

- If users do not want to use docker, you can refer to [Compile Environment Preparation for ARM Linux Environments](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/arm_linux_compile_arm_linux.rst) or [Compile Environment Preparation for X86 Linux Environments](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/linux_x86_compile_linux_x86.rst) provided by Paddle Lite and configure your own compilation environment, and then download and install the proper CANN packages to complete the configuration.

## C++ FastDeploy library compilation based on Paddle Lite
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
When the compilation is complete, the fastdeploy-ascend directory is created in the current build directory, indicating that the FastDeploy library has been compiled.

## Compiling Python FastDeploy Libraries Based on Paddle Lite

```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_ASCEND=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel

#After the compilation is complete, please install the whl package in the dist folder of the current directory.
```

## Enable FlyCV for Ascend deployment

[FlyCV](https://github.com/PaddlePaddle/FlyCV) is a high performance computer image processing library, providing better performance than other image processing libraries, especially in the ARM architecture.
FastDeploy is now integrated with FlyCV, allowing users to use FlyCV on supported hardware platforms to accelerate model end-to-end inference performance.
In end-to-end model inference, the pre-processing and post-processing phases are CPU computation, we recommend using FlyCV for end-to-end inference performance acceleration when you are using ARM CPU + Ascend hardware platform. See [Enable FlyCV](./boost_cv_by_flycv.md) documentation for details.


## Deployment demo reference
| Model | C++ Example | Python Example |
| :-----------| :--------   | :--------------- |
|   PaddleClas       |   [Ascend NPU C++ Example](../../../examples/vision/classification/paddleclas/cpp/README.md)       |    [Ascend NPU Python Example](../../../examples/vision/classification/paddleclas/python/README.md)          |  
|   PaddleDetection  |      [Ascend NPU C++ Example](../../../examples/vision/detection/paddledetection/cpp/README.md)        |     [Ascend NPU Python Example](../../../examples/vision/detection/paddledetection/python/README.md)               |
|   PaddleSeg        |      [Ascend NPU C++ Example](../../../examples/vision/segmentation/paddleseg/cpp/README.md)        |      [Ascend NPU Python Example](../../../examples//vision/segmentation/paddleseg/python/README.md)              |
|   PaddleOCR        |     [Ascend NPU C++ Example](../../../examples/vision/ocr/PP-OCRv3/cpp/README.md)         |      [Ascend NPU Python Example](../../../examples/vision//ocr/PP-OCRv3/python/README.md)              |
|   Yolov5           |      [Ascend NPU C++ Example](../../../examples/vision/detection/yolov5/cpp/README.md)       |       [Ascend NPU Python Example](../../../examples/vision/detection/yolov5/python/README.md)             |
|   Yolov6           |      [Ascend NPU C++ Example](../../../examples/vision/detection/yolov6/cpp/README.md)        |       [Ascend NPU Python Example](../../../examples/vision/detection/yolov6/python/README.md)             |
|   Yolov7           |      [Ascend NPU C++ Example](../../../examples/vision/detection/yolov7/cpp/README.md)        |       [Ascend NPU Python Example](../../../examples/vision/detection/yolov7/python/README.md)             |
