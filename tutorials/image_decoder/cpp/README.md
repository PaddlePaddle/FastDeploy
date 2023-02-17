English | [中文](README_CN.md)

# Image Decoder C++ Example

1. [Build FastDeploy](../docs/cn/build_and_install) or download [FastDeploy prebuilt library](../docs/cn/build_and_install/download_prebuilt_libraries.md)

2. Build example
```bash
mkdir build
cd build

# [PATH-TO-FASTDEPLOY] is the install directory of FastDeploy
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-TO-FASTDEPLOY]
make -j

# Download the test image
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# OpenCV decoder
./image_decoder ILSVRC2012_val_00000010.jpeg 0
# nvJPEG
./image_decoder ILSVRC2012_val_00000010.jpeg 1
