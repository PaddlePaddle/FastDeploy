[English](README.md) | 简体中文
# RKYOLO C++部署示例

本目录下提供`infer_xxxxx.cc`快速完成RKYOLO模型在Rockchip板子上上通过二代NPU加速部署的示例。

在部署前，需确认以下两个步骤:

1. 软硬件环境满足要求
2. 根据开发环境，下载预编译部署库或者从头编译FastDeploy仓库

以上步骤请参考[RK2代NPU部署库编译](../../../../../docs/cn/build_and_install/rknpu2.md)实现

```bash
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j8
./infer_rkyolo /path/to/model 000000014439.jpg
```


- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
