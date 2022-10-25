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

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_classification_PaddleClasModel_bindNative(
    JNIEnv *env, jclass clazz, jstring model_file, jstring params_file,
    jstring config_file, jint cpu_num_thread, jboolean enable_lite_fp16,
    jint lite_power_mode, jstring lite_optimized_model_dir,
    jboolean enable_record_time_of_runtime, jstring label_file) {
  std::string c_model_file =
      fastdeploy::jni::ConvertTo<std::string>(env, model_file);
  std::string c_params_file =
      fastdeploy::jni::ConvertTo<std::string>(env, params_file);
  std::string c_config_file =
      fastdeploy::jni::ConvertTo<std::string>(env, config_file);
  std::string c_label_file =
      fastdeploy::jni::ConvertTo<std::string>(env, label_file);
  std::string c_lite_optimized_model_dir =
      fastdeploy::jni::ConvertTo<std::string>(env, lite_optimized_model_dir);
  auto c_cpu_num_thread = static_cast<int>(cpu_num_thread);
  auto c_enable_lite_fp16 = static_cast<bool>(enable_lite_fp16);
  auto c_lite_power_mode =
      static_cast<fastdeploy::LitePowerMode>(lite_power_mode);
  fastdeploy::RuntimeOption c_option;
  c_option.UseCpu();
  c_option.UseLiteBackend();
  c_option.SetCpuThreadNum(c_cpu_num_thread);
  c_option.SetLitePowerMode(c_lite_power_mode);
  c_option.SetLiteOptimizedModelDir(c_lite_optimized_model_dir);
  if (c_enable_lite_fp16) {
    c_option.EnableLiteFP16();
  }
  auto c_model_ptr = new fastdeploy::vision::classification::PaddleClasModel(
      c_model_file, c_params_file, c_config_file, c_option);
  // Enable record Runtime time costs.
  if (enable_record_time_of_runtime) {
    c_model_ptr->EnableRecordTimeOfRuntime();
  }
  // Load classification labels if label path is not empty.
  if ((!fastdeploy::jni::AssetsLoaderUtils::IsClassificationLabelsLoaded()) &&
      (!c_label_file.empty())) {
    fastdeploy::jni::AssetsLoaderUtils::LoadClassificationLabels(c_label_file);
  }
  // WARN: need to release manually in Java !
  return reinterpret_cast<jlong>(c_model_ptr);  // native model context
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_classification_PaddleClasModel_predictNative(
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
  auto c_model_ptr =
      reinterpret_cast<fastdeploy::vision::classification::PaddleClasModel *>(
          native_model_context);
  auto c_result_ptr = new fastdeploy::vision::ClassifyResult();
  t = fastdeploy::jni::GetCurrentTime();
  if (!c_model_ptr->Predict(&c_bgr, c_result_ptr, 100)) {
    delete c_result_ptr;
    return 0;
  }
  LOGD("Predict from native costs %f ms", fastdeploy::jni::GetElapsedTime(t));
  if (c_model_ptr->EnabledRecordTimeOfRuntime()) {
    auto info_of_runtime = c_model_ptr->PrintStatisInfoOfRuntime();
    LOGD("Avg runtime costs %f ms", info_of_runtime["avg_time"] * 1000.0f);
  }
  if (!c_result_ptr->scores.empty() && rendering) {
    t = fastdeploy::jni::GetCurrentTime();
    cv::Mat c_vis_im;
    if (fastdeploy::jni::AssetsLoaderUtils::IsClassificationLabelsLoaded()) {
      c_vis_im = fastdeploy::vision::VisClassification(
          c_bgr, *(c_result_ptr),
          fastdeploy::jni::AssetsLoaderUtils::GetClassificationLabels(), 5,
          score_threshold, 1.0f);
    } else {
      c_vis_im = fastdeploy::vision::VisClassification(
          c_bgr, *(c_result_ptr), 5, score_threshold, 1.0f);
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
      cv::imwrite(c_saved_image_path, c_bgr);
      LOGD("Save image from native costs %f ms, path: %s",
           fastdeploy::jni::GetElapsedTime(t), c_saved_image_path.c_str());
    }
  }
  // WARN: need to release it manually in Java !
  return reinterpret_cast<jlong>(c_result_ptr);  // native result context
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_classification_PaddleClasModel_releaseNative(
    JNIEnv *env, jclass clazz, jlong native_model_context) {
  auto c_model_ptr =
      reinterpret_cast<fastdeploy::vision::classification::PaddleClasModel *>(
          native_model_context);
  if (c_model_ptr->EnabledRecordTimeOfRuntime()) {
    auto info_of_runtime = c_model_ptr->PrintStatisInfoOfRuntime();
    LOGD("[End] Avg runtime costs %f ms",
         info_of_runtime["avg_time"] * 1000.0f);
  }
  delete c_model_ptr;
  LOGD("[End] Release PaddleClasModel in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif
