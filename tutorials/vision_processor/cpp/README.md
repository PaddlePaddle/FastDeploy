English | [中文](README_CN.md)

# Preprocessor Python Demo

1. Compile FastDeploy and open CV-CUDA option
    > [Compile FastDeploy](../../../docs/cn/build_and_install/gpu.md)  
    > [Open CV-CUDA option](../../../docs/cn/faq/use_cv_cuda.md)

2. Run the demo
```bash
# Download the test image
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Compile the Demo
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../build/compiled_fastdeploy_sdk/ # if build sdk in `FastDeploy/build/compiled_fastdeploy_sdk`
make -j

# Run the demo

# Use OpenCV
./preprocessor_demo ILSVRC2012_val_00000010.jpeg 0

# Use CV-CUDA
./preprocessor_demo ILSVRC2012_val_00000010.jpeg 1
```
