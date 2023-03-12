English | [中文](README_CN.md)

# Preprocessor Python Demo

1. [build FastDeploy（Python）](../../../docs/cn/build_and_install), or download[FastDeploy prebuilt library（Python）](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

2. Run the Demo
```bash
# Download the test image
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Run the Demo

# OpenCV
python preprocess.py

# CV-CUDA
python preprocess.py --use_cvcuda True
```
