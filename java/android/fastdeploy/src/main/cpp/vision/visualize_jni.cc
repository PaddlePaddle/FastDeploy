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
#include "vision/results_jni.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

/// VisClassification
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visClassificationNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result, jfloat score_threshold, jfloat font_size,
    jobjectArray labels) {
  // Allocate cxx result
  fastdeploy::vision::ClassifyResult c_result;
  if(!fastdeploy::jni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      fastdeploy::vision::ResultType::CLASSIFY)) {
    return JNI_FALSE;
  }
  // Get labels from Java [n]
  auto c_labels = fastdeploy::jni::ConvertTo<std::vector<std::string>>(env, labels);

  cv::Mat c_bgr;
  // From ARGB Bitmap to BGR
  if (!fastdeploy::jni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  cv::Mat c_vis_im;
  if (!c_labels.empty()) {
    c_vis_im = fastdeploy::vision::VisClassification(
        c_bgr, c_result, c_labels, 5, score_threshold, font_size);
  } else {
    c_vis_im = fastdeploy::vision::VisClassification(
        c_bgr, c_result, 5, score_threshold, font_size);
  }
  // Rendering to bitmap
  if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

/// VisDetection
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visDetectionNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap, jobject result,
    jfloat score_threshold, jint line_size, jfloat font_size,
    jobjectArray labels) {
  // // Allocate cxx result
  fastdeploy::vision::DetectionResult c_result;
  if(!fastdeploy::jni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      fastdeploy::vision::ResultType::DETECTION)) {
    return JNI_FALSE;
  }

  // Get labels from Java [n]
  auto c_labels = fastdeploy::jni::ConvertTo<std::vector<std::string>>(env, labels);

  cv::Mat c_bgr;
  // From ARGB Bitmap to BGR
  if (!fastdeploy::jni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  cv::Mat c_vis_im;
  if (!c_labels.empty()) {
    c_vis_im = fastdeploy::vision::VisDetection(
        c_bgr, c_result, c_labels, score_threshold, line_size, font_size);
  } else {
    c_vis_im = fastdeploy::vision::VisDetection(
        c_bgr, c_result, score_threshold, line_size, font_size);
  }
  // Rendering to bitmap
  if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

/// VisOcr
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visOcrNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result) {
  // Allocate cxx result
  fastdeploy::vision::OCRResult c_result;
  if(!fastdeploy::jni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      fastdeploy::vision::ResultType::OCR)) {
    return JNI_FALSE;
  }

  cv::Mat c_bgr;
  // From ARGB Bitmap to BGR
  if (!fastdeploy::jni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  auto c_vis_im = fastdeploy::vision::VisOcr(c_bgr, c_result);
  // Rendering to bitmap
  if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif

