[English](README.md) | 简体中文

# PaddleDetection C++部署示例

本目录下提供`infer_picodet.cc`快速完成PPDetection模型在Rockchip板子上上通过二代NPU加速部署的示例。

在部署前，需确认以下两个步骤:

1. 软硬件环境满足要求
2. 根据开发环境，下载预编译部署库或者从头编译FastDeploy仓库

以上步骤请参考[RK2代NPU部署库编译](../../../../../../docs/cn/build_and_install/rknpu2.md)实现

```bash
# 以picodet为例进行推理部署

mkdir build
cd build
# 下载预编译库，详情见文档导航处
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/picodet_s_416_coco_lcnet.zip
unzip picodet_s_416_coco_lcnet.zip
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU推理
./infer_picodet_demo ./picodet_s_416_coco_lcnet 000000014439.jpg 0
# RKNPU2推理
./infer_picodet_demo ./picodet_s_416_coco_lcnet 000000014439.jpg 1
```

## 文档导航

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../../docs/api/vision_results/)
- [RKNPU2 预编译库](../../../../../../docs/cn/faq/rknpu2/rknpu2.md)
