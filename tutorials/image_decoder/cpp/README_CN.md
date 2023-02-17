简体中文 | [English](README.md)

# Image Decoder C++示例

1. [编译FastDeploy](../docs/cn/build_and_install), 或直接下载[FastDeploy预编译库](../docs/cn/build_and_install/download_prebuilt_libraries.md)

2. 编译示例
```bash
mkdir build
cd build

# [PATH-TO-FASTDEPLOY]需替换为FastDeploy的安装路径
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-TO-FASTDEPLOY]
make -j

# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# OpenCV解码
./image_decoder ILSVRC2012_val_00000010.jpeg 0
# nvJPEG
./image_decoder ILSVRC2012_val_00000010.jpeg 1
