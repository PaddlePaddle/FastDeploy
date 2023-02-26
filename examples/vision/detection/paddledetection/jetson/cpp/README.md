English | [简体中文](README_CN.md)
# PaddleDetection C++ Deployment Example

This directory provides examples that `infer_xxx.cc` fast finishes the deployment of PaddleDetection models, including SOLOv2 on CPU/GPU and GPU accelerated by TensorRT.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

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
