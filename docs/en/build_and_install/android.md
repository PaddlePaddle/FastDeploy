English | [中文](../../cn/build_and_install/android.md)

# How to Build FastDeploy Android C++ SDK

FastDeploy supports Paddle Lite backend on Android. It supports both armeabi-v7a and arm64-v8a cpu architectures, and supports fp16 precision inference on the armv8.2 architecture. The relevant compilation options are described as follows:

|Option|Default|Description|Remark|  
|:---|:---|:---|:---|  
|ENABLE_LITE_BACKEND|OFF|It needs to be set to ON when compiling the Android library| - |
|WITH_OPENCV_STATIC|OFF|Whether to use the OpenCV static library| - |
|WITH_LITE_STATIC|OFF|Whether to use the Paddle Lite static library| NOT Support now |

Please reference [FastDeploy Compile Options](./README.md) for more details.

## Build Android C++ SDK

Prerequisite for Compiling on Android:  

- Android SDK API >= 21  
- Android NDK >= 20 (Only support clang toolchain now)
- cmake >= 3.10.0  

Please check if the Android SDK and NDK is ready or not before building：  
```bash
➜ echo $ANDROID_SDK  
/Users/xxx/Library/Android/sdk  
➜ echo $ANDROID_NDK
/Users/xxx/Library/Android/sdk/ndk/25.1.8937393
```
It is recommended to use NDK>=20 for cross compilation, the compilation command is as follows：
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
      -DCMAKE_INSTALL_PREFIX=${FASDEPLOY_INSTALL_DIR} \
      -Wno-dev ../../..

# Build FastDeploy Android C++ SDK
make -j8
make install  
```  
After the compilation is complete, the Android C++ SDK is saved in the `build/Android/arm64-v8a-api-21/install` directory, the directory structure is as follows：  
```bash
➜ tree . -d -L 3
.
├── examples
├── include
│   └── fastdeploy                   # FastDeploy headers
├── lib
│   └── arm64-v8a                    # FastDeploy Android libs
└── third_libs                       # Third parties libs
    └── install
        ├── opencv
        ├── flycv
        └── paddlelite
```
You can check the Android C++ SDK use cases in the examples/vision directory：
```bash  
.
├── classification
│   ├── paddleclas
│   │   ├── android                  #  classification demo for Android
│   │   ├── cpp
...
├── detection
│   ├── paddledetection
│   │   ├── android                  #  object detection demo for Android
│   │   ├── cpp
...
```
About How to use FastDeploy Android C++ SDK, Please refer to the use case documentation:  
- [Image Classification Android Documentation](../../../examples/vision/classification/paddleclas/android/README.md)  
- [Object Detection Android Documentation](../../../examples/vision/detection/paddledetection/android/README.md)  
- [Using FastDeploy C++ SDK in Android via JNI](../../en/faq/use_cpp_sdk_on_android.md)
