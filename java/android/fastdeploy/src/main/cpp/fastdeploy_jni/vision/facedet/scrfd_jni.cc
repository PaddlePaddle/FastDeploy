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
#include "fastdeploy_jni/runtime_option_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/facedet/facedet_utils_jni.h"  // NOLINT

namespace fni = fastdeploy::jni;
namespace vision = fastdeploy::vision;
namespace facedet = fastdeploy::vision::facedet;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_facedet_SCRFD_bindNative(
    JNIEnv *env, jobject thiz, jstring model_file, jstring params_file,
    jobject runtime_option) {
  auto c_model_file = fni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = fni::ConvertTo<std::string>(env, params_file);
  auto c_runtime_option = fni::NewCxxRuntimeOption(env, runtime_option);
  auto c_model_ptr = new facedet::SCRFD(
      c_model_file, c_params_file, c_runtime_option, fastdeploy::ModelFormat::PADDLE);
  INITIALIZED_OR_RETURN(c_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
  c_model_ptr->EnableRecordTimeOfRuntime();
#endif
  // Setup input size, such as (320, 320), H x W
  const jclass j_scrfd_clazz = env->GetObjectClass(thiz);
  const jfieldID j_scrfd_size_id = env->GetFieldID(
      j_scrfd_clazz, "mSize", "[I");
  jintArray j_scrfd_size = reinterpret_cast<jintArray>(
      env->GetObjectField(thiz, j_scrfd_size_id));
  const auto c_size = fni::ConvertTo<std::vector<int>>(env, j_scrfd_size);
  c_model_ptr->size = c_size; // e.g (320, 320)
  env->DeleteLocalRef(j_scrfd_clazz); // release local Refs

  vision::EnableFlyCV();
  return reinterpret_cast<jlong>(c_model_ptr);
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_vision_facedet_SCRFD_predictNative(
    JNIEnv *env, jobject thiz, jlong cxx_context,
    jobject argb8888_bitmap, jfloat conf_threshold,
    jfloat nms_iou_threshold, jboolean save_image,
    jstring save_path, jboolean rendering) {
  if (cxx_context == 0) {
    return NULL;
  }
  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return NULL;
  }
  auto c_model_ptr = reinterpret_cast<facedet::SCRFD *>(cxx_context);
  vision::FaceDetectionResult c_result;
  auto t = fni::GetCurrentTime();
  c_model_ptr->Predict(&c_bgr, &c_result, conf_threshold, nms_iou_threshold);
  PERF_TIME_OF_RUNTIME(c_model_ptr, t)
  if (rendering) {
    fni::RenderingFaceDetection(env, c_bgr, c_result, argb8888_bitmap,
                                save_image, save_path);
  }
  return fni::NewJavaResultFromCxx(env, reinterpret_cast<void *>(&c_result),
                                   vision::ResultType::FACE_DETECTION);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_facedet_SCRFD_releaseNative(
    JNIEnv *env, jobject thiz, jlong cxx_context) {
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<facedet::SCRFD *>(cxx_context);
  PERF_TIME_OF_RUNTIME(c_model_ptr, -1)

  delete c_model_ptr;
  LOGD("[End] Release SCRFD in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif