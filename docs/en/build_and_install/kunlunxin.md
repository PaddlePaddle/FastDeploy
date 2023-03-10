English | [中文](../../cn/build_and_install/kunlunxin.md)

# How to Build KunlunXin XPU Deployment Environment

FastDeploy supports deployment AI on KunlunXin XPU based on Paddle Lite backend. For more detailed information, please refer to: [Paddle Lite Deployment Example](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/kunlunxin_xpu.html#xpu).

This document describes how to compile the C++ FastDeploy library based on Paddle Lite.

The relevant compilation options are described as follows:  
|Compile Options|Default Values|Description|Remarks|  
|:---|:---|:---|:---|  
| ENABLE_LITE_BACKEND | OFF | It needs to be set to ON when compiling the KunlunXin XPU library| - |  
| WITH_KUNLUNXIN | OFF | It needs to be set to ON when compiling the KunlunXin XPU library| - |
| ENABLE_VISION | OFF | whether to intergrate vision models | - |
| ENABLE_TEXT | OFF | whether to intergrate text models | - |

The configuration for third libraries(Optional, if the following option is not defined, the prebuilt third libraries will download automaticly while building FastDeploy).
| Option                     | Description                                                                                           |
| :---------------------- | :--------------------------------------------------------------------------------------------- |
| OPENCV_DIRECTORY        | While ENABLE_VISION=ON, use OPENCV_DIRECTORY to specify your own OpenCV library path.     |

For more compilation options, please refer to [Description of FastDeploy compilation options](./README.md)

## C++ FastDeploy library compilation based on Paddle Lite
- OS: Linux
- gcc/g++: version >= 8.2
- cmake: version >= 3.15

It it recommend install OpenCV library manually, and define `-DOPENCV_DIRECTORY` to set path of OpenCV library(If the flag is not defined, a prebuilt OpenCV library will be downloaded automaticly while building FastDeploy, but the prebuilt OpenCV cannot support reading video file or other function e.g `imshow`)
```
sudo apt-get install libopencv-dev
```

The compilation command is as follows:
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# CMake configuration with KunlunXin xpu toolchain
cmake -DWITH_KUNLUNXIN=ON  \
      -DWITH_GPU=OFF  \
      -DCMAKE_INSTALL_PREFIX=fastdeploy-kunlunxin \
      -DENABLE_VISION=ON \
      -DOPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
      ..

# Build FastDeploy KunlunXin XPU C++ SDK
make -j8
make install
```  
After the compilation is complete, the fastdeploy-kunlunxin directory will be generated, indicating that the Padddle Lite based FastDeploy library has been compiled.

## Python compile
The compilation command is as follows:
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export WITH_KUNLUNXIN=ON
export WITH_GPU=OFF
export ENABLE_VISION=ON
# The OPENCV_DIRECTORY is optional, if not exported, a prebuilt OpenCV library will be downloaded
export OPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4

python setup.py build
python setup.py bdist_wheel
```  
After the compilation is completed, the compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory, just pip install it directly

During the compilation process, if you modify the compilation parameters, in order to avoid the cache impact, you can delete the two subdirectories `build` and `.setuptools-cmake-build` under the `FastDeploy/python` directory and then recompile.
