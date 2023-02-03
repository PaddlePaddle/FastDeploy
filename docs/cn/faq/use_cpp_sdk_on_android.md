
[English](../../en/faq/use_cpp_sdk_on_android.md) | 中文

# 在 Android 中通过 JNI 使用 FastDeploy C++ SDK  
本文档将以PicoDet为例，讲解如何通过JNI，将FastDeploy中的模型封装到Android中进行调用。阅读本文档，您至少需要了解C++、Java、JNI以及Android的基础知识。如果您主要关注如何在Java层如何调用FastDeploy的API，则可以不阅读本文档。

## 目录  
- [在 Android 中通过 JNI 使用 FastDeploy C++ SDK](#在-android-中通过-jni-使用-fastdeploy-c-sdk)
  - [目录](#目录)
  - [新建Java类并定义native API](#新建java类并定义native-api)
  - [Android Studio 生成JNI函数定义](#android-studio-生成jni函数定义)
  - [在C++层实现JNI函数](#在c层实现jni函数)
  - [编写CMakeLists.txt及配置build.gradle](#编写cmakeliststxt及配置buildgradle)
  - [更多FastDeploy Android 使用案例](#更多fastdeploy-android-使用案例)

## 新建Java类并定义native API  
<div id="Java"></div>

```java  
public class PicoDet {
    protected long mNativeModelContext = 0; // Context from native.
    protected boolean mInitialized = false;
    // ...
    // Bind predictor from native context.
    private static native long bindNative(String modelFile,
                                          String paramsFile,
                                          String configFile,
                                          int cpuNumThread,
                                          boolean enableLiteFp16,
                                          int litePowerMode,
                                          String liteOptimizedModelDir,
                                          boolean enableRecordTimeOfRuntime,
                                          String labelFile);

    // Call prediction from native context.
    private static native long predictNative(long nativeModelContext,
                                             Bitmap ARGB8888Bitmap,
                                             boolean saved,
                                             String savedImagePath,
                                             float scoreThreshold,
                                             boolean rendering);

    // Release buffers allocated in native context.
    private static native boolean releaseNative(long nativeModelContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
```
这些被标记为native的接口是需要通过JNI的方式实现，并在Java层供PicoDet类调用。完整的PicoDet Java代码请参考 [PicoDet.java](../../../java/android/fastdeploy/src/main/java/com/baidu/paddle/fastdeploy/vision/detection/PicoDet.java) 。各个函数说明如下：  
- `bindNative`: C++层初始化模型资源，如果成功初始化，则返回指向该模型的指针(long类型)，否则返回0指针  
- `predictNative`: 通过已经初始化好的模型指针，在C++层执行预测代码，如果预测成功则返回指向预测结果的指针，否则返回0指针。注意，该结果指针在当次预测使用完之后需要释放，具体操作请参考 [PicoDet.java](../../../java/android/fastdeploy/src/main/java/com/baidu/paddle/fastdeploy/vision/detection/PicoDet.java) 中的predict函数。  
- `releaseNative`: 根据传入的模型指针，在C++层释放模型资源。

## Android Studio 生成JNI函数定义  
<div id="JNI"></div>

Android Studio 生成 JNI 函数定义: 鼠标停留在Java中定义的native函数上，Android Studio 便会提示是否要创建JNI函数定义；这里，我们把JNI函数定义创建在一个事先创建好的c++文件`picodet_jni.cc`上;

- 使用Android Studio创建JNI函数定义：
![](https://user-images.githubusercontent.com/31974251/197341065-cdf8f626-4bb1-4a57-8d7a-80b382fe994e.png)  

- 将JNI函数定义创建在picodet_jni.cc上：
![](https://user-images.githubusercontent.com/31974251/197341190-b887dec5-fa75-43c9-9ab3-7ead50c0eb45.png)

- 创建的JNI函数定义如下：  
![](https://user-images.githubusercontent.com/31974251/197341274-e9671bac-9e77-4043-a870-9d5db914586b.png)

其他native函数对应的JNI函数定义的创建和此流程一样。  

## 在C++层实现JNI函数  
<div id="CPP"></div>

以下为PicoDet JNI层实现的示例，相关的辅助函数不在此处赘述，完整的C++代码请参考 [android/app/src/main/cpp](../../examples/vision/detection/paddledetection/android/app/src/main/cpp/).
```C++
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>  // NOLINT
#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/assets_loader_jni.h" // NOLINT
#include "fastdeploy_jni/runtime_option_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/detection/detection_utils_jni.h"  // NOLINT

namespace fni = fastdeploy::jni;
namespace vision = fastdeploy::vision;
namespace detection = fastdeploy::vision::detection;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_bindNative(
    JNIEnv *env, jobject thiz, jstring model_file, jstring params_file,
    jstring config_file, jobject runtime_option, jstring label_file) {
  auto c_model_file = fni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = fni::ConvertTo<std::string>(env, params_file);
  auto c_config_file = fni::ConvertTo<std::string>(env, config_file);
  auto c_label_file = fni::ConvertTo<std::string>(env, label_file);
  auto c_runtime_option = fni::NewCxxRuntimeOption(env, runtime_option);
  auto c_model_ptr = new detection::PicoDet(
      c_model_file, c_params_file, c_config_file, c_runtime_option);
  INITIALIZED_OR_RETURN(c_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
  c_model_ptr->EnableRecordTimeOfRuntime();
#endif
  if (!c_label_file.empty()) {
    fni::AssetsLoader::LoadDetectionLabels(c_label_file);
  }
  vision::EnableFlyCV();
  return reinterpret_cast<jlong>(c_model_ptr);
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_predictNative(
    JNIEnv *env, jobject thiz, jlong cxx_context, jobject argb8888_bitmap,
    jboolean save_image, jstring save_path, jboolean rendering,
    jfloat score_threshold) {
  if (cxx_context == 0) {
    return NULL;
  }
  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return NULL;
  }
  auto c_model_ptr = reinterpret_cast<detection::PicoDet *>(cxx_context);
  vision::DetectionResult c_result;
  auto t = fni::GetCurrentTime();
  c_model_ptr->Predict(&c_bgr, &c_result);
  PERF_TIME_OF_RUNTIME(c_model_ptr, t)

  if (rendering) {
    fni::RenderingDetection(env, c_bgr, c_result, argb8888_bitmap, save_image,
                            score_threshold, save_path);
  }

  return fni::NewJavaResultFromCxx(env, reinterpret_cast<void *>(&c_result),
                                   vision::ResultType::DETECTION);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_releaseNative(
    JNIEnv *env, jobject thiz, jlong cxx_context) {
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<detection::PicoDet *>(cxx_context);
  PERF_TIME_OF_RUNTIME(c_model_ptr, -1)

  delete c_model_ptr;
  LOGD("[End] Release PicoDet in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif
```  
## 编写CMakeLists.txt及配置build.gradle  
<div id="CMakeAndGradle"></div>  

实现好的JNI代码，需要被编译成so库，才能被Java调用，为实现该目的，需要在build.gradle中添加JNI项目支持，并编写对应的CMakeLists.txt。  
- build.gradle中配置NDK、CMake以及Android ABI  
```java
android {
    defaultConfig {
        // 省略其他配置 ...
        externalNativeBuild {
            cmake {
                arguments '-DANDROID_PLATFORM=android-21', '-DANDROID_STL=c++_shared', "-DANDROID_TOOLCHAIN=clang"
                abiFilters 'armeabi-v7a', 'arm64-v8a'
                cppFlags "-std=c++11"
            }
        }
    }
    // 省略其他配置 ...
    externalNativeBuild {
        cmake {
            path file('src/main/cpp/CMakeLists.txt')
            version '3.10.2'
        }
    }
    sourceSets {
        main {
            jniLibs.srcDirs = ['libs']
        }
    }
    ndkVersion '20.1.5948944'
}
```  
- 编写CMakeLists.txt示例  
```cmake  
cmake_minimum_required(VERSION 3.10.2)
project("fastdeploy_jni")

# 其中 xxx 表示对应C++ SDK的版本号
set(FastDeploy_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/fastdeploy-android-xxx-shared")

find_package(FastDeploy REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${FastDeploy_INCLUDE_DIRS})

add_library(
        fastdeploy_jni
        SHARED
        utils_jni.cc
        bitmap_jni.cc
        vision/results_jni.cc
        vision/visualize_jni.cc
        vision/detection/picodet_jni.cc
        vision/classification/paddleclas_model_jni.cc)

find_library(log-lib log)

target_link_libraries(
        # Specifies the target library.
        fastdeploy_jni
        jnigraphics
        ${FASTDEPLOY_LIBS}
        GLESv2
        EGL
        ${log-lib}
)
```
完整的工程示例，请参考 [CMakelists.txt](../../../java/android/fastdeploy/src/main/cpp/CMakeLists.txt) 以及 [build.gradle](../../../java/android/fastdeploy/build.gradle).

## 更多FastDeploy Android 使用案例  
<div id="Examples"></div>  

更多FastDeploy Android 使用案例请参考以下文档:  
- [图像分类Android使用文档](../../../examples/vision/classification/paddleclas/android/README.md)  
- [目标检测Android使用文档](../../../examples/vision/detection/paddledetection/android/README.md)  
