# 昆仑芯 XPU 部署环境编译安装

FastDeploy 基于 Paddle-Lite 后端支持在昆仑芯 XPU 上进行部署推理。
更多详细的信息请参考：[PaddleLite部署示例](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/kunlunxin_xpu.html#xpu)。

本文档介绍如何编译基于 PaddleLite 的 C++ FastDeploy 编译库。

相关编译选项说明如下：  
|编译选项|默认值|说明|备注|  
|:---|:---|:---|:---|  
|ENABLE_LITE_BACKEND|OFF|编译RK库时需要设置为ON| - |

更多编译选项请参考[FastDeploy编译选项说明](./README.md)

## 基于 PaddleLite 的 FastDeploy 库编译
编译命令如下：
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

昆仑芯 XPU 上部署 PaddleClas 分类模型请参考：[PaddleClas 在昆仑芯 XPU 上的 C++ 部署示例](../../../examples/vision/classification/paddleclas/xpu/README.md)
