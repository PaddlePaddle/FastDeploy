# How to Build KunlunXin XPU Deployment Environment

FastDeploy supports deployment AI on KunlunXin XPU based on Paddle-Lite backend. For more detailed information, please refer to: [PaddleLite Deployment Example](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/kunlunxin_xpu.html#xpu)。

This document describes how to compile the C++ FastDeploy library based on PaddleLite.

The relevant compilation options are described as follows:  
|Compile Options|Default Values|Description|Remarks|  
|:---|:---|:---|:---|  
|ENABLE_LITE_BACKEND|OFF|It needs to be set to ON when compiling the RK library| - |  
|WITH_XPU|OFF|It needs to be set to ON when compiling the KunlunXin XPU library| - |  

For more compilation options, please refer to [Description of FastDeploy compilation options](./README.md)

## C++ FastDeploy library compilation based on PaddleLite
The compilation command is as follows:
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# CMake configuration with KunlunXin xpu toolchain
cmake -DWITH_XPU=ON  \
      -DCMAKE_INSTALL_PREFIX=fastdeploy-xpu \
      -DENABLE_VISION=ON \ # 是否编译集成视觉模型的部署模块，可选择开启
      ..

# Build FastDeploy KunlunXin XPU C++ SDK
make -j8
make install
```  
编译完成之后，会生成 fastdeploy-xpu 目录，表示基于 PadddleLite 的 FastDeploy 库编译完成。

## Python compile
The compilation command is as follows:
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_XPU=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel
```  

1. For deploying the PaddleClas classification model on Kunlun XPU, please refer to: [C++ deployment example of PaddleClas on Kunlun XPU](../../../examples/vision/classification/paddleclas/xpu/README.md)
2. For deploying YOLOv5 detection model on Kunlun Core XPU, please refer to: [C++ Deployment Example of YOLOv5 Detection Model on Kunlun Core XPU](../../../examples/vision/detection/yolov5/xpu/README.md)
