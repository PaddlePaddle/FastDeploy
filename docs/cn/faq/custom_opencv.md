[English](../../en/faq/custom_opencv.md) | 中文

# 自定义OpenCV版本

受限于不同平台限制，目前FastDeploy提供的预编译包在**Linux平台**内置的OpenCV无法读取视频，或调用`imshow`等操作。对于有这类需求的开发者，可根据本文档来自行编译FastDeploy。

FastDeploy目前支持通过`-DOPENCV_DIRECTORY`来指定环境中的OpenCV版本，以Ubuntu为例，我们可以按照如下方式编译安装。


## CPU C++ SDK

### 1. 安装Opencv
```
sudo apt-get install libopencv-dev
```

### 2. 指定OpenCV编译FastDeploy
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
编译完成的C++ SDK即为当前目录下的`installed_fastdeploy`，使用这个新的SDK即可。

其它部署硬件上的编译方式同理，通过`-DOPENCV_DIRECTORY`指定环境中的OpenCV编译即可, 注意此处的路径`/usr/lib/x86_64-linux-gnu/cmake/opencv4`需根据你的实际环境路径来设定，此目录下包含`OpenCVConfig-version.cmake`、`OpenCVConfig.cmake`等文件。

- [FastDeploy更多部署环境的编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/README_CN.md)
