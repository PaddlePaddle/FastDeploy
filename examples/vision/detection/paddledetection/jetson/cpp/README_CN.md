[English](README.md) | 简体中文
# PaddleDetection C++部署示例

本目录下提供`infer_xxx.cc`快速完成PaddleDetection模型包括SOLOv2在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)


```bash
mkdir build
cd build

# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
# CPU推理
./infer_solov2_demo ./solov2_r50_fpn_1x_coco 000000014439.jpg 0
# GPU推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 1
```
