# 在 Android 中通过 JNI 使用 FastDeploy C++ SDK  
本文档将以PicoDet为例，讲解如何通过JNI，将FastDeploy中的模型封装到Android中进行调用。阅读本文档，您至少需要了解C++、Java、JNI以及Android的基础知识。如果您主要关注如何在Java层如何调用FastDeploy的API，则可以不阅读本文档。

## 新建PicoDet Java类及定义需要C++层实现的native API  
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
这些被标记为native的接口是需要通过JNI的方式实现，并在Java层供PicoDet类调用。完整的PicoDet Java代码请参考 [PicoDet.java](../../../examples/vision/detection/paddledetection/android/app/src/main/java/com/baidu/paddle/fastdeploy/vision/detection/PicoDet.java) 。各个函数说明如下：  
- bindNative: C++层初始化模型资源，如果成功初始化，则返回指向该模型的指针(long类型)，否则返回0指针  
- predictNative: 通过已经初始化好的模型指针，在C++层执行预测代码，如果预测成功则返回指向预测结果的指针，否则返回0指针。注意，该结果指针在当次预测使用完之后需要释放，具体操作请参考 [PicoDet.java](../../../examples/vision/detection/paddledetection/android/app/src/main/java/com/baidu/paddle/fastdeploy/vision/detection/PicoDet.java) 中的predict函数。  
- releaseNative: 根据传入的模型指针，在C++层释放模型资源。

## Android Studio 生成JNI函数定义
- Android Studio 生成 JNI 函数定义: 鼠标停留在Java中定义的native函数上，Android Studio 便会提示是否要创建JNI函数定义；这里，我们把JNI函数定义创建在一个事先创建好的c++文件`picodet_jni.cc`上;

使用Android Studio创建JNI函数定义：
![](https://user-images.githubusercontent.com/31974251/197341065-cdf8f626-4bb1-4a57-8d7a-80b382fe994e.png)  

将JNI函数定义创建在picodet_jni.cc上：
![](https://user-images.githubusercontent.com/31974251/197341190-b887dec5-fa75-43c9-9ab3-7ead50c0eb45.png)

创建的JNI函数定义如下：  
![](https://user-images.githubusercontent.com/31974251/197341274-e9671bac-9e77-4043-a870-9d5db914586b.png)

其他native函数对应的JNI函数定义的创建和此流程一样。  

## 在C++层实现JNI函数  
以下为PicoDet JNI层实现的示例，相关的辅助函数不在此处赘述，完整的C++代码请参考 [android/app/src/main/cpp](../../../examples/vision/detection/paddledetection/android/app/src/main/cpp/).
```C++
#include <jni.h>  // NOLINT

#include "fastdeploy_jni.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

// 绑定C++层的模型
JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_bindNative(
    JNIEnv *env, jclass clazz, jstring model_file, jstring params_file,
    jstring config_file, jint cpu_num_thread, jboolean enable_lite_fp16,
    jint lite_power_mode, jstring lite_optimized_model_dir,
    jboolean enable_record_time_of_runtime, jstring label_file) {
  std::string c_model_file = fastdeploy::jni::ConvertTo<std::string>(env, model_file);
  std::string c_params_file = fastdeploy::jni::ConvertTo<std::string>(env, params_file);
  std::string c_config_file = astdeploy::jni::ConvertTo<std::string>(env, config_file);
  std::string c_label_file = fastdeploy::jni::ConvertTo<std::string>(env, label_file);
  std::string c_lite_optimized_model_dir = fastdeploy::jni::ConvertTo<std::string>(env, lite_optimized_model_dir);
  auto c_cpu_num_thread = static_cast<int>(cpu_num_thread);
  auto c_enable_lite_fp16 = static_cast<bool>(enable_lite_fp16);
  auto c_lite_power_mode = static_cast<fastdeploy::LitePowerMode>(lite_power_mode);
  fastdeploy::RuntimeOption c_option;
  c_option.UseCpu();
  c_option.UseLiteBackend();
  c_option.SetCpuThreadNum(c_cpu_num_thread);
  c_option.SetLitePowerMode(c_lite_power_mode);
  c_option.SetLiteOptimizedModelDir(c_lite_optimized_model_dir);
  if (c_enable_lite_fp16) {
    c_option.EnableLiteFP16();
  }
  // 如果您实现的是其他模型，比如PPYOLOE，请注意修改此处绑定的C++类型
  auto c_model_ptr = new fastdeploy::vision::detection::PicoDet(
      c_model_file, c_params_file, c_config_file, c_option);
  // Enable record Runtime time costs.
  if (enable_record_time_of_runtime) {
    c_model_ptr->EnableRecordTimeOfRuntime();
  }
  // Load detection labels if label path is not empty.
  if ((!fastdeploy::jni::AssetsLoaderUtils::IsDetectionLabelsLoaded()) &&
      (!c_label_file.empty())) {
    fastdeploy::jni::AssetsLoaderUtils::LoadDetectionLabels(c_label_file);
  }
  // WARN: need to release manually in Java !
  return reinterpret_cast<jlong>(c_model_ptr);  // native model context
}

// 通过传入的模型指针在C++层进行预测
JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_predictNative(
    JNIEnv *env, jclass clazz, jlong native_model_context,
    jobject argb8888_bitmap, jboolean saved, jstring saved_image_path,
    jfloat score_threshold, jboolean rendering) {
  if (native_model_context == 0) {
    return 0;
  }
  cv::Mat c_bgr;
  auto t = fastdeploy::jni::GetCurrentTime();
  if (!fastdeploy::jni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return 0;
  }
  LOGD("Read from bitmap costs %f ms", fastdeploy::jni::GetElapsedTime(t));
  auto c_model_ptr = reinterpret_cast<fastdeploy::vision::detection::PicoDet *>(
      native_model_context);
  auto c_result_ptr = new fastdeploy::vision::DetectionResult();
  t = fastdeploy::jni::GetCurrentTime();
  if (!c_model_ptr->Predict(&c_bgr, c_result_ptr)) {
    delete c_result_ptr;
    return 0;
  }
  LOGD("Predict from native costs %f ms", fastdeploy::jni::GetElapsedTime(t));
  if (c_model_ptr->EnabledRecordTimeOfRuntime()) {
    auto info_of_runtime = c_model_ptr->PrintStatisInfoOfRuntime();
    LOGD("Avg runtime costs %f ms", info_of_runtime["avg_time"] * 1000.0f);
  }
  if (!c_result_ptr->boxes.empty() && rendering) {
    t = fastdeploy::jni::GetCurrentTime();
    cv::Mat c_vis_im;
    if (fastdeploy::jni::AssetsLoaderUtils::IsDetectionLabelsLoaded()) {
      c_vis_im = fastdeploy::vision::VisDetection(
          c_bgr, *(c_result_ptr),
          fastdeploy::jni::AssetsLoaderUtils::GetDetectionLabels(),
          score_threshold, 2, 1.0f);
    } else {
      c_vis_im = fastdeploy::vision::VisDetection(c_bgr, *(c_result_ptr),
                                                  score_threshold, 2, 1.0f);
    }
    LOGD("Visualize from native costs %f ms",
         fastdeploy::jni::GetElapsedTime(t));
    // Rendering to bitmap
    t = fastdeploy::jni::GetCurrentTime();
    if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      delete c_result_ptr;
      return 0;
    }
    LOGD("Write to bitmap from native costs %f ms",
         fastdeploy::jni::GetElapsedTime(t));
    std::string c_saved_image_path =
        fastdeploy::jni::ConvertTo<std::string>(env, saved_image_path);
    if (!c_saved_image_path.empty() && saved) {
      t = fastdeploy::jni::GetCurrentTime();
      cv::imwrite(c_saved_image_path, c_vis_im);
      LOGD("Save image from native costs %f ms, path: %s",
           fastdeploy::jni::GetElapsedTime(t), c_saved_image_path.c_str());
    }
  }
  // WARN: need to release it manually in Java !
  return reinterpret_cast<jlong>(c_result_ptr);  // native result context
}

// 在C++层释放模型资源
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_releaseNative(
    JNIEnv *env, jclass clazz, jlong native_model_context) {
  if (native_model_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<fastdeploy::vision::detection::PicoDet *>(
      native_model_context);
  if (c_model_ptr->EnabledRecordTimeOfRuntime()) {
    auto info_of_runtime = c_model_ptr->PrintStatisInfoOfRuntime();
    LOGD("[End] Avg runtime costs %f ms",
         info_of_runtime["avg_time"] * 1000.0f);
  }
  delete c_model_ptr;
  LOGD("[End] Release PicoDet in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif
```  
## 编写CMakeLists.txt及配置build.gradle  
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
    ndkVersion '20.1.5948944'
}
```  
- 编写CMakeLists.txt示例  
```cmake  
cmake_minimum_required(VERSION 3.10.2)
project("fastdeploy_jni")

set(FastDeploy_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/fastdeploy-android-0.4.0-shared")

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
完整的工程示例，请参考 [android/app/src/main/cpp/CMakelists.txt](../../../examples/vision/detection/paddledetection/android/app/src/main/cpp/) 以及 [android/app/build.gradle](../../../examples/vision/detection/paddledetection/android/app/build.gradle).

## 更多FastDeploy Android 使用案例请参考以下文档：  
- [图像分类Android使用文档](../../../examples/vision/classification/paddleclas/android/README.md)  
- [目标检测Android使用文档](../../../examples/vision/detection/paddledetection/android/README.md)  
