[English](../../en/build_and_install/kunlunxin.md) | 简体中文

# 昆仑芯 XPU 部署环境编译安装

FastDeploy 基于 Paddle Lite 后端支持在昆仑芯 XPU 上进行部署推理。
更多详细的信息请参考：[Paddle Lite部署示例](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/kunlunxin_xpu.html#xpu)。

本文档介绍如何编译基于 Paddle Lite 的 C++ FastDeploy 编译库。

相关编译选项说明如下：  
|编译选项|默认值|说明|备注|  
|:---|:---|:---|:---|  
| WITH_KUNLUNXIN| OFF | 需要在昆仑芯 XPU 上部署时需要设置为 ON | - |
| ENABLE_VISION | OFF | 是否编译集成视觉模型的部署模块 | - |
| ENABLE_TEXT | OFF | 是否编译集成文本 NLP 模型的部署模块 | - |

第三方库依赖指定（不设定如下参数，会自动下载预编译库）
| 选项                     | 说明                                                                                           |
| :---------------------- | :--------------------------------------------------------------------------------------------- |
| OPENCV_DIRECTORY        | 当 ENABLE_VISION=ON 时，用于指定用户本地的 OpenCV 库路径；如果不指定，编译过程会自动下载 OpenCV 库              |

更多编译选项请参考[FastDeploy编译选项说明](./README.md)

## 基于 Paddle Lite 的 C++ FastDeploy 库编译
- OS: Linux
- gcc/g++: version >= 8.2
- cmake: version >= 3.15

此外更推荐开发者自行安装，编译时通过 `-DOPENCV_DIRECTORY` 来指定环境中的 OpenCV（如若不指定 -DOPENCV_DIRECTORY，会自动下载 FastDeploy 提供的预编译的 OpenCV，但在 **Linux平台** 无法支持 Video 的读取，以及 imshow 等可视化界面功能）

```
sudo apt-get install libopencv-dev
```
编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# CMake configuration with KunlunXin xpu toolchain
cmake -DWITH_KUNLUNXIN=ON  \
      -DWITH_GPU=OFF  \ # 不编译 GPU
      -DCMAKE_INSTALL_PREFIX=fastdeploy-kunlunxin \
      -DENABLE_VISION=ON \ # 是否编译集成视觉模型的部署模块，可选择开启
      -DOPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \ # 指定系统自带的 opencv 路径
      ..

# Build FastDeploy KunlunXin XPU C++ SDK
make -j8
make install
```  
编译完成之后，会生成 fastdeploy-kunlunxin 目录，表示基于 Paddle Lite 的 FastDeploy 库编译完成。

## Python 编译
编译命令如下：
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_KUNLUNXIN=ON
export WITH_GPU=OFF
export ENABLE_VISION=ON
# OPENCV_DIRECTORY 可选，不指定会自动下载 FastDeploy 提供的预编译 OpenCV 库
export OPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4

python setup.py build
python setup.py bdist_wheel
```  
编译完成即会在 `FastDeploy/python/dist` 目录下生成编译后的 `wheel` 包，直接 pip install 即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除 `FastDeploy/python` 目录下的 `build` 和 `.setuptools-cmake-build` 两个子目录后再重新编译
