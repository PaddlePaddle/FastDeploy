中文 | [English](README.md)

# Preprocessor Python 示例代码

1. [编译FastDeploy（Python）](../docs/cn/build_and_install), 或直接下载[FastDeploy预编译库（Python）](../docs/cn/build_and_install/download_prebuilt_libraries.md)

2. 运行示例代码
```bash
# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 运行示例代码

# OpenCV
python preprocess.py

# CV-CUDA
python preprocess.py --use_cvcuda True
```
