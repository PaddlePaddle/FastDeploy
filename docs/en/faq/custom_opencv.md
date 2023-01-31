English | [中文](../../cn/faq/custom_opencv.md)

# Use Own OpenCV Library

The prebuilt FastDeploy library has a built-in OpenCV library, which is not able to read video file or call `imshow` because the prebuilt FastDeploy has to build in manylinux version. If you need to read video or other functions provided by opencv, this document shows how to build FastDeploy with your own OpenCV in your environment.

FastDeploy provides flag `-DOPENCV_DIRECTORY` to set path of OpenCV library, the following steps show how to build CPU C++ SDK on Ubuntu.

## CPU C++ SDK

### 1. Install OpenCV

```
sudo apt-get install libopencv-dev
```

### 2. Build FastDeploy

```
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DENABLE_VISION=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/installed_fastdeploy \
         -DOPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4
make -j8
make install
```

Now we get the C++ SDK in current directory `installed_fastdeploy`, this library can use all the functions from your own OpenCV library.

This document also works for other hardware deployment(GPU/IPU/XPU...) on Linux platform. 

- [More Options to build FastDeploy](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/README_EN.md)
