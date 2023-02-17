# 华为昇腾NPU 部署环境编译准备

## 导航目录

* [简介以及编译选项](#简介以及编译选项)
* [华为昇腾环境准备](#一华为昇腾环境准备)
* [编译环境搭建](#二编译环境搭建)
* [基于 Paddle Lite 的 C++ FastDeploy 库编译](#三基于-paddle-lite-的-c-fastdeploy-库编译)
* [基于 Paddle Lite 的 Python FastDeploy 库编译](#四基于-paddle-lite-的-python-fastdeploy-库编译)
* [昇腾部署时开启FlyCV](#五昇腾部署时开启flycv)
* [昇腾部署Demo参考](#六昇腾部署demo参考)

## 简介以及编译选项

FastDeploy基于 Paddle-Lite 后端, 支持在华为昇腾NPU上进行部署推理。
更多详细的信息请参考：[Paddle Lite部署示例](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/huawei_ascend_npu.md)。

本文档介绍如何在ARM/X86_64的Linux操作系统环境下, 编译基于 Paddle Lite 的 C++ 与 Python 的FastDeploy源码, 生成目标硬件为华为昇腾NPU的预测库。

更多编译选项请参考[FastDeploy编译选项说明](./README.md)


## 一.华为昇腾环境准备
- Atlas 300I Pro 推理卡, 详情见[规格说明书](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300i-pro)
- 安装Atlas 300I Pro 推理卡的驱动和固件包（Driver 和 Firmware)
- 配套驱动和固件包下载：
  - https://www.hiascend.com/hardware/firmware-drivers?tag=community（社区版）
  - https://www.hiascend.com/hardware/firmware-drivers?tag=commercial（商业版）
  - 驱动：Atlas-300i-pro-npu-driver_5.1.rc2_linux-aarch64.run (这里以aarch64平台为例)
  - 固件：Atlas-300i-pro-npu-firmware_5.1.rc2.run
- 安装驱动和固件包：

```shell
# 增加可执行权限
$ chmod +x *.run
# 安装驱动和固件包
$ ./Atlas-300i-pro-npu-driver_5.1.rc2_linux-aarch64.run --full
$ ./Atlas-300i-pro-npu-firmware_5.1.rc2.run --full
# 重启服务器
$ reboot
# 查看驱动信息，确认安装成功
$ npu-smi info
```
- 更多系统和详细信息见[昇腾硬件产品文档](https://www.hiascend.com/document?tag=hardware)


## 二.编译环境搭建

### 宿主机环境需求  
- os：ARM-Linux, X86_64-Linux
- gcc、g++、git、make、wget、python、pip、python-dev、patchelf
- cmake（建议使用 3.10 或以上版本）

### 使用Docker开发环境
为了保证和FastDeploy验证过的编译环境一致，建议使用Docker开发环境进行配置.

aarch64平台示例
```shell
# 下载 Dockerfile
$ wget https://bj.bcebos.com/fastdeploy/test/Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile
# 通过 Dockerfile 生成镜像
$ docker build --network=host -f Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile -t paddlelite/ascend_aarch64:cann_5.1.rc2 .
# 创建容器
$ docker run -itd --privileged --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_aarch64:cann_5.1.rc2 /bin/bash
# 进入容器
$ docker exec -it ascend-aarch64 /bin/bash
# 确认容器的 Ascend 环境是否创建成功
$ npu-smi info
```

x86_64平台示例
```shell
# 下载 Dockerfile
$ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/Ascend_ubuntu18.04_x86_5.1.rc1.alpha001.Dockerfile
# 通过 Dockerfile 生成镜像
$ docker build --network=host -f Ascend_ubuntu18.04_x86_5.1.rc1.alpha001.Dockerfile -t paddlelite/ascend_x86:cann_5.1.rc1.alpha001 .
# 创建容器
$ docker run -itd --privileged --name=ascend-x86 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_x86:cann_5.1.1.alpha001 /bin/bash
# 进入容器
$ docker exec -it ascend-x86 /bin/bash
# 确认容器的 Ascend 环境是否创建成功
$ npu-smi info
```

以上步骤成功后，用户可以直接在docker内部开始FastDeploy的编译.

注意:
- 如果用户在Docker内想使用其他的CANN版本,请自行更新 Dockerfile 文件内的 CANN 下载路径, 同时更新相应的驱动和固件. 当前示例中, aarch64平台的Dockerfile内默认为[CANN 5.1.RC2](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%205.1.RC2/Ascend-cann-toolkit_5.1.RC2_linux-aarch64.run), x86_64平台的Dockerfile内默认为[CANN 5.1.RC1](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/5.1.RC1.alpha001/Ascend-cann-toolkit_5.1.RC1.alpha001_linux-x86_64.run).

- 如果用户不想使用docker，可以参考由Paddle Lite提供的[ARM Linux环境下的编译环境准备](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/arm_linux_compile_arm_linux.rst)或者[X86 Linux环境下的编译环境准备](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/linux_x86_compile_linux_x86.rst)自行配置编译环境, 之后再自行下载并安装相应的CANN软件包来完成配置.

## 三.基于 Paddle Lite 的 C++ FastDeploy 库编译
搭建好编译环境之后，编译命令如下：
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
编译完成之后，会在当前的build目录下生成 fastdeploy-ascend 目录，表示基于 Paddle Lite 的 FastDeploy 库编译完成。

## 四.基于 Paddle Lite 的 Python FastDeploy 库编译
搭建好编译环境之后，编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_ASCEND=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel

#编译完成后,请用户自行安装当前目录的dist文件夹内的whl包.
```
## 五.昇腾部署时开启FlyCV
[FlyCV](https://github.com/PaddlePaddle/FlyCV) 是一款高性能计算机图像处理库, 针对ARM架构做了很多优化, 相比其他图像处理库性能更为出色.
FastDeploy现在已经集成FlyCV, 用户可以在支持的硬件平台上使用FlyCV, 实现模型端到端推理性能的加速.
模型端到端推理中, 预处理和后处理阶段为CPU计算, 当用户使用ARM CPU + 昇腾的硬件平台时, 我们推荐用户使用FlyCV, 可以实现端到端的推理性能加速, 详见[FlyCV使用文档](../faq/boost_cv_by_flycv.md).


## 六.昇腾部署Demo参考

| 模型系列 | C++ 部署示例 | Python 部署示例 |
| :-----------| :--------   | :--------------- |
|   PaddleClas       |   [昇腾NPU C++ 部署示例](../../../examples/vision/classification/paddleclas/cpp/README_CN.md)       |    [昇腾NPU Python 部署示例](../../../examples/vision/classification/paddleclas/python/README_CN.md)          |  
|   PaddleDetection  |      [昇腾NPU C++ 部署示例](../../../examples/vision/detection/paddledetection/cpp/README_CN.md)        |     [昇腾NPU Python 部署示例](../../../examples/vision/detection/paddledetection/python/README_CN.md)               |
|   PaddleSeg        |      [昇腾NPU C++ 部署示例](../../../examples/vision/segmentation/paddleseg/cpp/README_CN.md)        |      [昇腾NPU Python 部署示例](../../../examples//vision/segmentation/paddleseg/python/README_CN.md)              |
|   PaddleOCR        |     [昇腾NPU C++ 部署示例](../../../examples/vision/ocr/PP-OCRv3/cpp/README_CN.md)         |      [昇腾NPU Python 部署示例](../../../examples/vision//ocr/PP-OCRv3/python/README_CN.md)              |
|   Yolov5           |      [昇腾NPU C++ 部署示例](../../../examples/vision/detection/yolov5/cpp/README_CN.md)       |       [昇腾NPU Python 部署示例](../../../examples/vision/detection/yolov5/python/README_CN.md)             |
|   Yolov6           |      [昇腾NPU C++ 部署示例](../../../examples/vision/detection/yolov6/cpp/README_CN.md)        |       [昇腾NPU Python 部署示例](../../../examples/vision/detection/yolov6/python/README_CN.md)             |
|   Yolov7           |      [昇腾NPU C++ 部署示例](../../../examples/vision/detection/yolov7/cpp/README_CN.md)        |       [昇腾NPU Python 部署示例](../../../examples/vision/detection/yolov7/python/README_CN.md)             |
