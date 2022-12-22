English | [中文](../../cn/faq/use_cpp_sdk_on_android.md)
# FastDeploy to deploy on Android Platform

This document will take PicoDet as an example and explain how to encapsulate FastDeploy model to Android through JNI. You need to know at least the basics of C++, Java, JNI and Android. If you mainly focus on how to call FastDeploy API in Java layer, you can skip this document.

## Content 
- [FastDeploy to deploy on Android Platform](#fastdeploy-to-deploy-on-android-platform)
  - [Content](#content)
  - [Create a new Java class and Define the native API](#create-a-new-java-class-and-define-the-native-api)
  - [Generate JNI function definition with Android Studio](#generate-jni-function-definition-with-android-studio)
  - [Implement JNI function in the C++ layer](#implement-jni-function-in-the-c-layer)
  - [Write CMakeLists.txt and configure build.gradle](#write-cmakeliststxt-and-configure-buildgradle)
  - [More examples of FastDeploy Android](#more-examples-of-fastdeploy-android)


## Create a new Java class and Define the native API
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
These interfaces, marked as native, are required to be implemented by JNI and should be available to call for Class PicoDet in the Java layer. For the complete PicoDet Java code, please refer to [PicoDet.java](../../../java/android/fastdeploy/src/main/java/com/baidu/paddle/fastdeploy/vision/detection/PicoDet.java). The functions are described seperately:
- `bindNative`: Initialize the model resource in the C++ layer. It returns a cursor (of type long) to the model if it is successfully initialized, otherwise it returns a 0 cursor.
- `predictNative`: Run the prediction code in th C++ layer with the initialized model cursor. If executed successfully, it returns a cursor to the result, otherwise it returns a 0 cursor. Please note that the cursor needs to be released after the current prediction, please refer to the definition of the `predict` funtion in [PicoDet.java](../../../java/android/fastdeploy/src/main/java/com/baidu/paddle/fastdeploy/vision/detection/PicoDet.java) for details.
- `releaseNative`: Release model resources in the C++ layer according to the input model cursor.

## Generate JNI function definition with Android Studio
<div id="JNI"></div>

 Hover over the native function defined in Java and Android Studio will prompt if you want to create a JNI function definition. Here, we create the definition in a pre-created c++ file `picodet_jni.cc`.

- Create a JNI function definition with Android Studio:
![](https://user-images.githubusercontent.com/31974251/197341065-cdf8f626-4bb1-4a57-8d7a-80b382fe994e.png)  

- Create the definition in picodet_jni.cc:
![](https://user-images.githubusercontent.com/31974251/197341190-b887dec5-fa75-43c9-9ab3-7ead50c0eb45.png)

- The JNI function definition created:
![](https://user-images.githubusercontent.com/31974251/197341274-e9671bac-9e77-4043-a870-9d5db914586b.png)

You can create JNI function definitions corresponding to other native functions referring to this process.

## Implement JNI function in the C++ layer
<div id="CPP"></div>

Here is an example of the PicoDet JNI layer implementation. For the complete C++ code, please refer to [android/app/src/main/cpp](../../../examples/vision/detection/paddledetection/android/app/src/main/cpp/).
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
## Write CMakeLists.txt and configure build.gradle  
<div id="CMakeAndGradle"></div>  

The implemented JNI code needs to be compiled into a so library to be called by Java. To achieve this, you need to add JNI project support in build.gradle, and write the corresponding CMakeLists.txt.
- Configure NDK, CMake and Android ABI in build.gradle
```java
android {
    defaultConfig {
        // Other configurations are omitted ...
        externalNativeBuild {
            cmake {
                arguments '-DANDROID_PLATFORM=android-21', '-DANDROID_STL=c++_shared', "-DANDROID_TOOLCHAIN=clang"
                abiFilters 'armeabi-v7a', 'arm64-v8a'
                cppFlags "-std=c++11"
            }
        }
    }
    // Other configurations are omitted ...
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
- An example of CMakeLists.txt
```cmake  
cmake_minimum_required(VERSION 3.10.2)
project("fastdeploy_jni")

# Where xxx indicates the version number of C++ SDK
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
For the complete project, please refer to [CMakelists.txt](../../../java/android/fastdeploy/src/main/cpp/CMakeLists.txt) and [build.gradle](../../../java/android/fastdeploy/build.gradle).

## More examples of FastDeploy Android
<div id="Examples"></div>  

For more examples of using FastDeploy Android, you can refer to:  
- [Image classification on Android](../../../examples/vision/classification/paddleclas/android/README.md)  
- [Object detection on Android](../../../examples/vision/detection/paddledetection/android/README.md)  

