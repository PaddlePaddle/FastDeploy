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

#include <android/bitmap.h>  // NOLINT
#include <jni.h>             // NOLINT

#include "fastdeploy/vision.h"  // NOLINT
#include "fastdeploy_jni.h"     // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

/// Native DetectionResult for vision::DetectionResult.
JNIEXPORT jint JNICALL
Java_com_baidu_paddle_fastdeploy_vision_DetectionResult_copyBoxesNumFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return 0;
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::DetectionResult *>(
      native_result_context);
  return static_cast<jint>(c_result_ptr->boxes.size());
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_DetectionResult_copyBoxesFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::DetectionResult *>(
      native_result_context);
  if (c_result_ptr->boxes.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->boxes.size());
  float buffer[len * 4];
  const auto &boxes = c_result_ptr->boxes;
  for (int64_t i = 0; i < len; ++i) {
    std::memcpy((buffer + i * 4), (boxes.at(i).data()), 4 * sizeof(float));
  }
  return fastdeploy::jni::ConvertTo<jfloatArray>(env, buffer, len * 4);
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_DetectionResult_copyScoresFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::DetectionResult *>(
      native_result_context);
  if (c_result_ptr->scores.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->scores.size());
  const float *buffer = static_cast<float *>(c_result_ptr->scores.data());
  return fastdeploy::jni::ConvertTo<jfloatArray>(env, buffer, len);
}

JNIEXPORT jintArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_DetectionResult_copyLabelIdsFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::DetectionResult *>(
      native_result_context);
  if (c_result_ptr->label_ids.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->label_ids.size());
  const int *buffer = static_cast<int *>(c_result_ptr->label_ids.data());
  return fastdeploy::jni::ConvertTo<jintArray>(env, buffer, len);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_DetectionResult_releaseNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return JNI_FALSE;
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::DetectionResult *>(
      native_result_context);
  delete c_result_ptr;
  LOGD("Release DetectionResult in native !");
  return JNI_TRUE;
}

/// Native ClassifyResult for vision::ClassifyResult.
JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_ClassifyResult_copyScoresFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::ClassifyResult *>(
      native_result_context);
  if (c_result_ptr->scores.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->scores.size());
  const float *buffer = static_cast<float *>(c_result_ptr->scores.data());
  return fastdeploy::jni::ConvertTo<jfloatArray>(env, buffer, len);
}

JNIEXPORT jintArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_ClassifyResult_copyLabelIdsFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::ClassifyResult *>(
      native_result_context);
  if (c_result_ptr->label_ids.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->label_ids.size());
  const int *buffer = static_cast<int *>(c_result_ptr->label_ids.data());
  return fastdeploy::jni::ConvertTo<jintArray>(env, buffer, len);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_ClassifyResult_releaseNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return JNI_FALSE;
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::ClassifyResult *>(
      native_result_context);
  delete c_result_ptr;
  LOGD("Release ClassifyResult in native !");
  return JNI_TRUE;
}

/// Native OCRResult for vision::OCRResult.
JNIEXPORT jint JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_copyBoxesNumFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return 0;
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  return static_cast<jint>(c_result_ptr->boxes.size());
}

JNIEXPORT jintArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_copyBoxesFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  if (c_result_ptr->boxes.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->boxes.size());
  int buffer[len * 8];
  const auto &boxes = c_result_ptr->boxes;
  for (int64_t i = 0; i < len; ++i) {
    std::memcpy((buffer + i * 8), (boxes.at(i).data()), 8 * sizeof(int));
  }
  return fastdeploy::jni::ConvertTo<jintArray>(env, buffer, len * 4);
}

JNIEXPORT jobjectArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_copyTextFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  if (c_result_ptr->text.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->text.size());
  jclass jstr_clazz = env->FindClass("java/lang/String");
  jobjectArray jstr_array = env->NewObjectArray(
      static_cast<jsize>(len), jstr_clazz,env->NewStringUTF(""));
  for (int64_t i = 0; i < len; ++i) {
    env->SetObjectArrayElement(jstr_array, static_cast<jsize>(i),
                               fastdeploy::jni::ConvertTo<jstring>(
                                   env, c_result_ptr->text.at(i)));
  }
  return jstr_array;
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_copyRecScoresFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  if (c_result_ptr->rec_scores.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->rec_scores.size());
  const float *buffer = static_cast<float *>(c_result_ptr->rec_scores.data());
  return fastdeploy::jni::ConvertTo<jfloatArray>(env, buffer, len);
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_copyClsScoresFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  if (c_result_ptr->cls_scores.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->cls_scores.size());
  const float *buffer = static_cast<float *>(c_result_ptr->cls_scores.data());
  return fastdeploy::jni::ConvertTo<jfloatArray>(env, buffer, len);
}

JNIEXPORT jintArray JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_copyClsLabelsFromNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return {};
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  if (c_result_ptr->cls_labels.empty()) {
    return {};
  }
  const auto len = static_cast<int64_t>(c_result_ptr->cls_labels.size());
  const int *buffer = static_cast<int *>(c_result_ptr->cls_labels.data());
  return fastdeploy::jni::ConvertTo<jintArray>(env, buffer, len);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_OCRResult_releaseNative(
    JNIEnv *env, jobject thiz, jlong native_result_context) {
  if (native_result_context == 0) {
    return JNI_FALSE;
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::vision::OCRResult *>(
      native_result_context);
  delete c_result_ptr;
  LOGD("Release OCRResult in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif

