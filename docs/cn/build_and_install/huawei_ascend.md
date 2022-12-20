# 华为昇腾NPU 部署环境编译准备

FastDeploy基于 Paddle-Lite 后端, 支持在华为昇腾NPU上进行部署推理。
更多详细的信息请参考：[Paddle Lite部署示例](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/huawei_ascend_npu.md)。

本文档介绍如何在ARM Linux操作系统环境下, 编译基于 Paddle Lite 的 C++ 与 Python 的FastDeploy源码, 生成目标硬件为华为昇腾NPU的预测库。

更多编译选项请参考[FastDeploy编译选项说明](./README.md)


## 一.华为昇腾环境准备
- Atlas 300I Pro 推理卡, 详情见[规格说明书](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-300i-pro)
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
$ wget https://bj.bcebos.com/fastdeploy/test/Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile
# 通过 Dockerfile 生成镜像
$ docker build --network=host -f Ascend_ubuntu18.04_aarch64_5.1.rc2.Dockerfile -t Paddle Lite/ascend_aarch64:cann_5.1.rc2 .
# 创建容器
$ docker run -itd --privileged --name=ascend-aarch64 --net=host -v $PWD:/Work -w /Work --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ Paddle Lite/ascend_aarch64:cann_5.1.rc2 /bin/bash
# 进入容器
$ docker exec -it ascend-aarch64 /bin/bash
# 确认容器的 Ascend 环境是否创建成功
$ npu-smi info
```
以上步骤成功后，用户可以直接在docker内部开始FastDeploy的编译.

注意:
- 如果用户在Docker内想使用其他的CANN版本,请自行更新 Dockerfile 文件内的 CANN 下载路径, 同时更新相应的驱动和固件. 当前Dockerfile内默认为[CANN 5.1.RC2](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%205.1.RC2/Ascend-cann-toolkit_5.1.RC2_linux-aarch64.run).
- 如果用户不想使用docker，可以参考由Paddle Lite提供的[ARM Linux环境下的编译环境准备](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/source_compile/arm_linux_compile_arm_linux.rst)自行配置编译环境, 之后再自行下载并安装相应的CANN软件包来完成配置.

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
编译完成之后，会在当前的build目录下生成 fastdeploy-ascend 目录，表示基于 PadddleLite 的 FastDeploy 库编译完成。

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

- 华为昇腾NPU 上使用C++部署 PaddleClas 分类模型请参考：[PaddleClas 华为升腾NPU C++ 部署示例](../../../examples/vision/classification/paddleclas/ascend/cpp/README.md)
- 华为昇腾NPU 上使用Python部署 PaddleClas 分类模型请参考：[PaddleClas 华为升腾NPU Python 部署示例](../../../examples/vision/classification/paddleclas/ascend/python/README.md)
