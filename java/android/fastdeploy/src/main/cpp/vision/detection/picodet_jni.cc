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

#include "fastdeploy_jni.h"  // NOLINT
#include "vision/results_jni.h" // NOLINT

namespace fastdeploy {
namespace jni {

/// Rendering DetectionResult to ARGB888Bitmap
void RenderingDetection(
    JNIEnv *env, const cv::Mat &c_bgr, const vision::DetectionResult &c_result,
    jobject argb8888_bitmap, bool saved, float score_threshold,
    jstring saved_image_path) {
  if (!c_result.boxes.empty()) {
    auto t = fastdeploy::jni::GetCurrentTime();
    cv::Mat c_vis_im;
    if (fastdeploy::jni::AssetsLoaderUtils::IsDetectionLabelsLoaded()) {
      c_vis_im = fastdeploy::vision::VisDetection(
          c_bgr, c_result,
          fastdeploy::jni::AssetsLoaderUtils::GetDetectionLabels(),
          score_threshold, 2, 1.0f);
    } else {
      c_vis_im = fastdeploy::vision::VisDetection(c_bgr, c_result,
                                                  score_threshold, 2, 1.0f);
    }
    LOGD("Visualize from native costs %f ms",
         fastdeploy::jni::GetElapsedTime(t));

    // Rendering to bitmap
    if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      LOGD("Write to bitmap from native failed!");
    }
    std::string c_saved_image_path =
        fastdeploy::jni::ConvertTo<std::string>(env, saved_image_path);
    if (!c_saved_image_path.empty() && saved) {
      cv::imwrite(c_saved_image_path, c_vis_im);
    }
  }
}

/// Show the time cost of Runtime
void PrintPicoDetTimeOfRuntime(
    vision::detection::PicoDet *c_model_ptr,
    int64_t start = -1) {
  if (c_model_ptr == nullptr) {
    return;
  }
  if (start > 0) {
    auto tc = fastdeploy::jni::GetElapsedTime(start);
    LOGD("Predict from native costs %f ms", tc);
  }
  if (c_model_ptr->EnabledRecordTimeOfRuntime()) {
    auto info_of_runtime = c_model_ptr->PrintStatisInfoOfRuntime();
    const float avg_time = info_of_runtime["avg_time"] * 1000.0f;
    LOGD("Avg runtime costs %f ms", avg_time);
  }
}

}  // namespace jni
}  // namespace fastdeploy

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_bindNative(
    JNIEnv *env, jobject thiz, jstring model_file, jstring params_file,
    jstring config_file, jobject runtime_option, jstring label_file) {
  auto c_model_file = fastdeploy::jni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = fastdeploy::jni::ConvertTo<std::string>(env, params_file);
  auto c_config_file = fastdeploy::jni::ConvertTo<std::string>(env, config_file);
  auto c_label_file = fastdeploy::jni::ConvertTo<std::string>(env, label_file);
  auto c_runtime_option = fastdeploy::jni::NewCxxRuntimeOption(
      env, runtime_option);
  auto c_model_ptr = new fastdeploy::vision::detection::PicoDet(
      c_model_file, c_params_file, c_config_file, c_runtime_option);
  // TODO(qiuyanjun): remove this flag
  c_model_ptr->EnableRecordTimeOfRuntime();
  if (!c_label_file.empty()) {
    fastdeploy::jni::AssetsLoaderUtils::LoadDetectionLabels(c_label_file);
  }
  // TODO(qiuyanjun): enable FlyCV according to proc_lib_option
  fastdeploy::vision::EnableFlyCV();
  return reinterpret_cast<jlong>(c_model_ptr);
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_predictNative(
    JNIEnv *env, jobject thiz, jlong native_model_context,
    jobject argb8888_bitmap, jboolean saved, jstring saved_image_path,
    jfloat score_threshold, jboolean rendering) {
  if (native_model_context == 0) {
    return NULL;
  }
  cv::Mat c_bgr;
  if (!fastdeploy::jni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return NULL;
  }
  auto c_model_ptr = reinterpret_cast<fastdeploy::vision::detection::PicoDet *>(
      native_model_context);
  fastdeploy::vision::DetectionResult c_result;
  auto t = fastdeploy::jni::GetCurrentTime();
  c_model_ptr->Predict(&c_bgr, &c_result);
  fastdeploy::jni::PrintPicoDetTimeOfRuntime(c_model_ptr, t);
  if (rendering) {
    fastdeploy::jni::RenderingDetection(env, c_bgr, c_result,
                                        argb8888_bitmap,saved,
                                        score_threshold, saved_image_path);
  }

  return fastdeploy::jni::NewJavaResultFromCxx(
      env, reinterpret_cast<void *>(&c_result),
      fastdeploy::vision::ResultType::DETECTION);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_detection_PicoDet_releaseNative(
    JNIEnv *env, jobject thiz, jlong native_model_context) {
  if (native_model_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<fastdeploy::vision::detection::PicoDet *>(
      native_model_context);
  fastdeploy::jni::PrintPicoDetTimeOfRuntime(c_model_ptr);
  delete c_model_ptr;
  LOGD("[End] Release PicoDet in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif
