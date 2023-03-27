中文 | [English](README.md)

# Preprocessor Python 示例代码

1. 编译FastDeploy并开启CV-CUDA选项
    > [编译FastDeploy](../../../docs/cn/build_and_install/gpu.md)  
    > [开启CV-CUDA选项](../../../docs/cn/faq/use_cv_cuda.md)

2. 运行示例代码
```bash
# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 编译示例代码
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../build/compiled_fastdeploy_sdk/ # 若编译FastDeploy在其他文件夹，请替换为相应的sdk路径
make -j

# 运行示例代码

# 使用OpenCV处理图片
./preprocessor_demo ILSVRC2012_val_00000010.jpeg 0

# 使用CV-CUDA处理图片
./preprocessor_demo ILSVRC2012_val_00000010.jpeg 1
```
