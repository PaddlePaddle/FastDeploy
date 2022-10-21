# Android部署库编译

FastDeploy当前在Android仅支持Paddle-Lite后端推理，支持armeabi-v7a和arm64-v8a两种cpu架构，在armv8.2架构的arm设备支持fp16精度推理。相关编译选项说明如下：  

|编译选项|默认值|说明|备注|  
|:---|:---|:---|:---|  
|ENABLE_LITE_BACKEND|OFF|编译Android库时需要设置为ON| - |
|WITH_OPENCV_STATIC|OFF|是否使用OpenCV静态库| - |
|WITH_LITE_STATIC|OFF|是否使用Lite静态库| 暂不支持使用Lite静态库 |
|WITH_LITE_FULL_API|ON|是否使用Lite Full API库| 目前必须为ON |
|WITH_LITE_FP16|ON|是否使用带FP16支持的Lite库| 目前仅支持 arm64-v8a 架构|

更多编译选项请参考[FastDeploy编译选项说明](./README.md)

## Android C++ SDK 编译安装  

编译需要满足：  

- Android SDK API >= 21  
- Android NDK >= 20 (当前仅支持clang编译工具链)
- cmake >= 3.10.0  

编译前请先检查您的Android SDK 和 NDK 是否已经配置，如：  
```bash
➜ echo $ANDROID_SDK  
/Users/xxx/Library/Android/sdk  
➜ echo $ANDROID_NDK
/Users/xxx/Library/Android/sdk/ndk/25.1.8937393
```
推荐使用 NDK>=20 进行交叉编译，编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  

# Setting up Android toolchanin
ANDROID_ABI=arm64-v8a  # 'arm64-v8a', 'armeabi-v7a'
ANDROID_PLATFORM="android-21"  # API >= 21
ANDROID_STL=c++_shared  # 'c++_shared', 'c++_static'
ANDROID_TOOLCHAIN=clang  # 'clang' only
TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake

# Create build directory
BUILD_ROOT=build/Android
BUILD_DIR=${BUILD_ROOT}/${ANDROID_ABI}-api-21
FASDEPLOY_INSTALL_DIR="${BUILD_DIR}/install"
mkdir build && mkdir ${BUILD_ROOT} && mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

# Check fp16 support (only support arm64-v8a now)
WITH_LITE_FP16=ON
if [ "$ANDROID_ABI" = "armeabi-v7a" ]; then
  WITH_LITE_FP16=OFF
fi

# CMake configuration with Android toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DANDROID_ABI=${ANDROID_ABI} \
      -DANDROID_NDK=${ANDROID_NDK} \
      -DANDROID_PLATFORM=${ANDROID_PLATFORM} \
      -DANDROID_STL=${ANDROID_STL} \
      -DANDROID_TOOLCHAIN=${ANDROID_TOOLCHAIN} \
      -DENABLE_LITE_BACKEND=ON \
      -DENABLE_VISION=ON \
      -DWITH_LITE_FP16=${WITH_LITE_FP16} \
      -DCMAKE_INSTALL_PREFIX=${FASDEPLOY_INSTALL_DIR} \
      -Wno-dev ../../..

# Build FastDeploy Android C++ SDK
make -j8
make install  
```  
编译完成后，Android C++ SDK 保存在 `build/Android/arm64-v8a-api-21/install` 目录下，目录结构如下：  
```bash
➜ tree . -d -L 3
.
├── examples
├── include
│   └── fastdeploy                   # FastDeploy 头文件
├── lib
│   └── arm64-v8a                    # FastDeploy Android 动态库
└── third_libs                       # 第三方依赖库
    └── install
        ├── opencv
        └── paddlelite
```
在examples/vision目录下可查看Android C++ SDK 使用案例：
```bash  
.
├── classification
│   ├── paddleclas
│   │   ├── android                  #  图像分类Android使用案例
│   │   ├── cpp
...
├── detection
│   ├── paddledetection
│   │   ├── android                  #  目标检测Android使用案例
│   │   ├── cpp
...
```
如何使用FastDeploy Android C++ SDK 请参考使用案例文档：  
- [图像分类Android使用文档](../../../examples/vision/classification/paddleclas/android/README.md)  
- [目标检测Android使用文档](../../../examples/vision/detection/paddledetection/android/README.md)  
