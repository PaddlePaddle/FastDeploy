# 华为昇腾NPU 部署环境编译准备

FastDeploy基于 Paddle-Lite 后端支持在华为昇腾NPU上进行部署推理。
更多详细的信息请参考：[PaddleLite部署示例](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/huawei_ascend_npu.md)。

本文档介绍如何在ARM Linux操作系统环境下, 编译基于 PaddleLite 的 C++ FastDeploy 源码, 生成目标硬件为华为昇腾NPU的预测库。

更多编译选项请参考[FastDeploy编译选项说明](./README.md)


## 一.设备环境的准备
- Atlas 300I Pro 推理卡[规格说明书](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300i-pro)
- 安装Atlas 300I Pro 推理卡的驱动和固件包（Driver 和 Firmware)
- 配套驱动和固件包下载：
  - https://www.hiascend.com/hardware/firmware-drivers?tag=community（社区版）
  - https://www.hiascend.com/hardware/firmware-drivers?tag=commercial（商业版）
  - 驱动：Atlas-300i-pro-npu-driver_5.1.rc2_linux-aarch64.run
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
- os：ARM-Linux
- gcc、g++、git、make、wget、python、pip、python-dev、patchelf
- cmake（建议使用 3.10 或以上版本）

### 使用Docker开发环境
为了保证和FastDeploy验证过的编译环境一致，建议使用Docker开发环境进行配置.

```shell
# 下载 Dockerfile
$ wget https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/kunpeng920_arm/Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile
# 通过 Dockerfile 生成镜像
$ docker build --network=host -f Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile -t paddlelite/ascend_aarch64:cann_5.1.rc2 .
# 创建容器
$ docker run -itd --privileged --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ paddlelite/ascend_aarch64:cann_5.1.1.alpha001 /bin/bash
# 进入容器
$ docker exec -it ascend-aarch64 /bin/bash
# 确认容器的 Ascend 环境是否创建成功
$ npu-smi info
```
以上步骤成功后，用户可以直接在docker内部开始FastDeploy的编译.

注意:
- 如果用户在Docker内想使用其他的CANN版本,请自行更新 Dockerfile 文件内的 CANN 下载路径, 同时更新相应的驱动和固件. 当前Dockerfile内默认为[CANN 5.1.RC2](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%205.1.RC2/Ascend-cann-toolkit_5.1.RC2_linux-aarch64.run).
- 如果用户不想使用docker，可以参考由PaddleLite提供的[ARM Linux环境下的编译环境准备](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/arm_linux_compile_arm_linux.rst)自行配置编译环境, 之后再自行下载并安装相应的CANN软件包来完成配置.

## 三.基于 PaddleLite 的 FastDeploy 库编译
搭建好编译环境之后，编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# CMake configuration with CANN
cmake -DCMAKE_TOOLCHAIN_FILE=./../cmake/cann.cmake \
      -DENABLE_CANN=ON  \
      -DCMAKE_INSTALL_PREFIX=fastdeploy-cann \
      -DENABLE_VISION=ON \ # 是否编译集成视觉模型的部署模块，可选择开启
      -Wno-dev ..

# Build FastDeploy CANN C++ SDK
make -j8
make install
```  
编译完成之后，会在当前的build目录下生成 fastdeploy-cann 目录，表示基于 PadddleLite CANN 的 FastDeploy 库编译完成。

华为昇腾NPU 上部署 PaddleClas 分类模型请参考：[PaddleClas 华为升腾NPU C++ 部署示例](../../../examples/vision/classification/paddleclas/ascend/README.md)
