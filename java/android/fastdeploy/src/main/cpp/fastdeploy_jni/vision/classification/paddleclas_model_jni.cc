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
#include "fastdeploy_jni/convert_jni.h"         // NOLINT
#include "fastdeploy_jni/assets_loader_jni.h"   // NOLINT
#include "fastdeploy_jni/runtime_option_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/classification/classification_utils_jni.h"  // NOLINT

namespace fni = fastdeploy::jni;
namespace vision = fastdeploy::vision;
namespace classification = fastdeploy::vision::classification;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_classification_PaddleClasModel_bindNative(
    JNIEnv *env, jobject thiz, jstring model_file, jstring params_file,
    jstring config_file, jobject runtime_option, jstring label_file) {
  auto c_model_file = fni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = fni::ConvertTo<std::string>(env, params_file);
  auto c_config_file = fni::ConvertTo<std::string>(env, config_file);
  auto c_label_file = fni::ConvertTo<std::string>(env, label_file);
  auto c_runtime_option = fni::NewCxxRuntimeOption(env, runtime_option);
  auto c_model_ptr = new classification::PaddleClasModel(
      c_model_file, c_params_file, c_config_file, c_runtime_option);
  INITIALIZED_OR_RETURN(c_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
  c_model_ptr->EnableRecordTimeOfRuntime();
#endif
  if (!c_label_file.empty()) {
    fni::AssetsLoader::LoadClassificationLabels(c_label_file);
  }
  vision::EnableFlyCV();
  return reinterpret_cast<jlong>(c_model_ptr);  // native model context
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_vision_classification_PaddleClasModel_predictNative(
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
  auto c_model_ptr =
      reinterpret_cast<classification::PaddleClasModel *>(cxx_context);

  vision::ClassifyResult c_result;
  auto t = fni::GetCurrentTime();
  c_model_ptr->Predict(&c_bgr, &c_result);
  PERF_TIME_OF_RUNTIME(c_model_ptr, t)

  if (rendering) {
    fni::RenderingClassify(env, c_bgr, c_result, argb8888_bitmap, save_image,
                           score_threshold, save_path);
  }

  return fni::NewJavaResultFromCxx(env, reinterpret_cast<void *>(&c_result),
                                   vision::ResultType::CLASSIFY);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_classification_PaddleClasModel_releaseNative(
    JNIEnv *env, jobject thiz, jlong cxx_context) {
  auto c_model_ptr =
      reinterpret_cast<classification::PaddleClasModel *>(cxx_context);
  PERF_TIME_OF_RUNTIME(c_model_ptr, -1)
  delete c_model_ptr;
  LOGD("[End] Release PaddleClasModel in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif
