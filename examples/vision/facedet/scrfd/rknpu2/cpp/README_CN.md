[English](README.md) | 简体中文
# SCRFD C++部署示例

本目录下提供`infer.cc`在RK356X上，快速完成SCRFD在NPU加速部署的示例。

在部署前，需确认以下两个步骤:

1. 软硬件环境满足要求
2. 根据开发环境，下载预编译部署库或者从头编译FastDeploy仓库

以上步骤请参考[RK2代NPU文档导航](../../../../../../docs/cn/build_and_install/rknpu2.md)实现

## 编译

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-rk356X-x.x.x.tgz
tar -xzvf fastdeploy-linux-aarch64-rk356X-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-aarch64-rk356X
make -j8
```

## 运行例程

```bash
#下载官方转换好的SCRFD模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/scrfd_500m_bnkps_shape640x640_rknpu2.zip
unzip scrfd_500m_bnkps_shape640x640_rknpu2.zip
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg
./infer_demo scrfd_500m_bnkps_shape640x640_rknpu2/scrfd_500m_bnkps_shape640x640_rk3568_quantized.rknn \
              test_lite_face_detector_3.jpg \
              1
```
运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301789-1981d065-208f-4a6b-857c-9a0f9a63e0b1.jpg">

- [模型介绍](../../README.md)
- [Python部署](../python/README.md)
- [视觉模型预测结果](../../../../../../docs/api/vision_results/README.md)
