English | [简体中文](../../cn/build_and_install/horizon.md)

# How to Build Horizon Deployment Environment

Horizon refers to the BPU of the Sunrise X3 series chips launched by Horizon. Currently, FastDeploy has initially supported deploying models using Horizon. If you encounter any problems during use, please provide your operating environment and feedback in the Issues section.

## Introduction

If you want to use the Horizon inference engine in FastDeploy, you need to configure the following environments:

| Tools                | Yes/No                   | Platform | Description                                         |
|:------------------|:---------------------|:-------|---------------------------------|
| Paddle2ONNX  | Yes   | PC    | Used to convert PaddleInference models to ONNX models    |  
| Horizon Environments Docker | Yes   | PC    | Used to convert ONNX models to Horizon models               |  
| Horizon XJ3 OpenExplorer       | Yes   | PC | header files and dynamic libraries |

## Model Conversion Environment

Horizon provides a complete model transformation environment (XJ3 chip toolchain image), and FastDeploy adopts the image version of
[2.5.2](ftp://vrftp.horizon.ai/Open_Explorer_gcc_9.3.0/2.5.2/docker_openexplorer_ubuntu_20_xj3_gpu_v2.5.2_py38.tar.gz), You can obtain it through the Horizon developer platform.



## Software Package

Horizon also provides a complete toolkit (Horizon XJ3 OpenExplorer)
, The development package version used by FastDeploy is
[2.5.2](ftp://vrftp.horizon.ai/Open_Explorer_gcc_9.3.0/2.5.2/horizon_xj3_openexplorer_v2.5.2_py38_20230331.tar.gz),  You can obtain it through the Horizon developer platform.

Due to the weak performance of the board CPU, it is recommended to perform cross compilation on a PC. The following tutorial is completed in the Docker environment provided by Horizon.



### Start Docker Environment
After downloading the Horizon XJ3 chip toolchain image locally, execute the following command to import the image package into the Docker environment:



```bash
docker load < docker_openexplorer_ubuntu_20_xj3_gpu_v2.5.2_py38.tar.gz
```
After downloading the dependent software packages to the local machine, unzip them:
```bash
tar -xvf horizon_xj3_openexplorer_v2.5.2_py38_20230331.tar.gz
```
After the unzipping is complete, cd to that directory:
```bash
cd horizon_xj3_open_explorer_v2.5.2-py38_20230331/
```

Under the root directory, there is a script to run Docker. Run the following command:
```bash
sh run_docker.sh /home gpu
```

The first directory is the directory to be mounted on the container, and the latter parameter is to enable GPU acceleration for the Docker.

At this point, the preparation of the required environment for compilation is complete.

## How to Build and Install C++ SDK
Download the cross-compilation tool, [gcc_linaro_6.5.0_2018.12_x86_64_aarch64_linux_gnu](https://bj.bcebos.com/fastdeploy/third_libs/gcc_linaro_6.5.0_2018.12_x86_64_aarch64_linux_gnu.tar.xz), and it is recommended to extract it to the `/opt` directory.
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

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


