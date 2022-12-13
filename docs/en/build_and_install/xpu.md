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
      -DWITH_GPU=OFF  \
      -DENABLE_ORT_BACKEND=ON  \
      -DENABLE_PADDLE_BACKEND=ON  \
      -DCMAKE_INSTALL_PREFIX=fastdeploy-xpu \
      -DENABLE_VISION=ON \
      ..

# Build FastDeploy KunlunXin XPU C++ SDK
make -j8
make install
```  
After the compilation is complete, the fastdeploy-xpu directory will be generated, indicating that the PadddleLite-based FastDeploy library has been compiled.

## Python compile
The compilation command is as follows:
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_XPU=ON
export WITH_GPU=OFF
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel
```  
After the compilation is completed, the compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory, just pip install it directly

During the compilation process, if you modify the compilation parameters, in order to avoid the cache impact, you can delete the two subdirectories `build` and `.setuptools-cmake-build` under the `FastDeploy/python` directory and then recompile.
