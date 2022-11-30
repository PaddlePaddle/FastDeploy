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
#include "fastdeploy_jni/perf_jni.h"  // NOLINT
#include "fastdeploy_jni/bitmap_jni.h"  // NOLINT
#include "fastdeploy_jni/convert_jni.h" // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT

namespace fni = fastdeploy::jni;
namespace vision = fastdeploy::vision;

namespace fastdeploy {
namespace jni {

/// Some visualize helpers.
jboolean VisClassificationFromJava(
    JNIEnv *env, jobject argb8888_bitmap,
    jobject result, jfloat score_threshold, jfloat font_size,
    jobjectArray labels) {
  const jclass j_cls_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/ClassifyResult");
  if (!env->IsInstanceOf(result, j_cls_result_clazz)) {
    env->DeleteLocalRef(j_cls_result_clazz);
    return JNI_FALSE;
  }
  env->DeleteLocalRef(j_cls_result_clazz);
  vision::ClassifyResult c_result;
  if (!fni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      vision::ResultType::CLASSIFY)) {
    return JNI_FALSE;
  }
  // Get labels from Java [n]
  auto c_labels = fni::ConvertTo<std::vector<std::string>>(env, labels);

  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  cv::Mat c_vis_im;
  if (!c_labels.empty()) {
    c_vis_im = vision::VisClassification(c_bgr, c_result, c_labels, 5,
                                         score_threshold, font_size);
  } else {
    c_vis_im = vision::VisClassification(c_bgr, c_result, 5, score_threshold,
                                         font_size);
  }
  if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean VisDetectionFromJava(
    JNIEnv *env, jobject argb8888_bitmap,
    jobject result, jfloat score_threshold, jint line_size,
    jfloat font_size, jobjectArray labels) {
  const jclass j_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/DetectionResult");
  if (!env->IsInstanceOf(result, j_det_result_clazz)) {
    env->DeleteLocalRef(j_det_result_clazz);
    return JNI_FALSE;
  }
  env->DeleteLocalRef(j_det_result_clazz);
  vision::DetectionResult c_result;
  if (!fni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      vision::ResultType::DETECTION)) {
    return JNI_FALSE;
  }
  // Get labels from Java [n]
  auto c_labels = fni::ConvertTo<std::vector<std::string>>(env, labels);

  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  cv::Mat c_vis_im;
  if (!c_labels.empty()) {
    c_vis_im = vision::VisDetection(c_bgr, c_result, c_labels, score_threshold,
                                    line_size, font_size);
  } else {
    c_vis_im = vision::VisDetection(c_bgr, c_result, score_threshold, line_size,
                                    font_size);
  }
  if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean VisOcrFromJava(
    JNIEnv *env, jobject argb8888_bitmap, jobject result) {
  const jclass j_ocr_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/OCRResult");
  if (!env->IsInstanceOf(result, j_ocr_result_clazz)) {
    env->DeleteLocalRef(j_ocr_result_clazz);
    return JNI_FALSE;
  }
  env->DeleteLocalRef(j_ocr_result_clazz);
  vision::OCRResult c_result;
  if (!fni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      vision::ResultType::OCR)) {
    return JNI_FALSE;
  }

  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  auto c_vis_im = vision::VisOcr(c_bgr, c_result);
  if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean VisSegmentationFromJava(
    JNIEnv *env, jobject argb8888_bitmap, jobject result, jfloat weight) {
  const jclass j_seg_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/SegmentationResult");
  if (!env->IsInstanceOf(result, j_seg_result_clazz)) {
    env->DeleteLocalRef(j_seg_result_clazz);
    return JNI_FALSE;
  }
  env->DeleteLocalRef(j_seg_result_clazz);
  // Allocate from Java result, may cost some times.
  vision::SegmentationResult c_result;
  if (!fni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      vision::ResultType::SEGMENTATION)) {
    return JNI_FALSE;
  }
  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  auto c_vis_im = vision::VisSegmentation(c_bgr, c_result, weight);
  if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean VisSegmentationFromCxxBuffer(
    JNIEnv *env, jobject argb8888_bitmap, jobject result, jfloat weight) {
  const jclass j_seg_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/SegmentationResult");
  if (!env->IsInstanceOf(result, j_seg_result_clazz)) {
    env->DeleteLocalRef(j_seg_result_clazz);
    return JNI_FALSE;
  }
  const jfieldID j_enable_cxx_buffer_id = env->GetFieldID(
      j_seg_result_clazz, "mEnableCxxBuffer", "Z");
  const jfieldID  j_cxx_buffer_id = env->GetFieldID(
      j_seg_result_clazz, "mCxxBuffer", "J");
  const jfieldID j_seg_initialized_id = env->GetFieldID(
      j_seg_result_clazz, "mInitialized", "Z");
  jboolean j_enable_cxx_buffer =
      env->GetBooleanField(result, j_enable_cxx_buffer_id);
  jboolean j_seg_initialized =
      env->GetBooleanField(result, j_seg_initialized_id);

  env->DeleteLocalRef(j_seg_result_clazz);
  if (j_seg_initialized == JNI_FALSE) {
    return JNI_FALSE;
  }
  // Use CxxBuffer directly without any copy.
  if (j_enable_cxx_buffer == JNI_TRUE) {
    jlong j_cxx_buffer = env->GetLongField(result, j_cxx_buffer_id);
    if (j_cxx_buffer == 0) {
      return JNI_FALSE;
    }
    // Allocate from cxx context to cxx result
    auto c_cxx_buffer = reinterpret_cast<vision::SegmentationResult *>(j_cxx_buffer);
    cv::Mat c_bgr;
    if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
      return JNI_FALSE;
    }
    auto c_vis_im = vision::VisSegmentation(c_bgr, *c_cxx_buffer, weight);
    if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      return JNI_FALSE;
    }
    return JNI_TRUE;
  }
  return JNI_FALSE;
}

jboolean VisFaceDetectionFromJava(
    JNIEnv *env, jobject argb8888_bitmap,
    jobject result, jint line_size, jfloat font_size) {
  const jclass j_face_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/FaceDetectionResult");
  if (!env->IsInstanceOf(result, j_face_det_result_clazz)) {
    env->DeleteLocalRef(j_face_det_result_clazz);
    return JNI_FALSE;
  }
  env->DeleteLocalRef(j_face_det_result_clazz);
  vision::FaceDetectionResult c_result;
  if (!fni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      vision::ResultType::FACE_DETECTION)) {
    return JNI_FALSE;
  }
  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  auto c_vis_im = vision::VisFaceDetection(c_bgr, c_result, line_size, font_size);
  if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean VisKeyPointDetectionFromJava(
    JNIEnv *env, jobject argb8888_bitmap, jobject result,
    jfloat conf_threshold) {
  const jclass j_keypoint_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/KeyPointDetectionResult");
  if (!env->IsInstanceOf(result, j_keypoint_det_result_clazz)) {
    env->DeleteLocalRef(j_keypoint_det_result_clazz);
    return JNI_FALSE;
  }
  env->DeleteLocalRef(j_keypoint_det_result_clazz);
  vision::KeyPointDetectionResult c_result;
  if (!fni::AllocateCxxResultFromJava(
      env, result, reinterpret_cast<void *>(&c_result),
      vision::ResultType::KEYPOINT_DETECTION)) {
    return JNI_FALSE;
  }
  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return JNI_FALSE;
  }
  auto c_vis_im = vision::VisKeypointDetection(c_bgr, c_result, conf_threshold);
  if (!fni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

}  // jni
}  // fastdeploy

#ifdef __cplusplus
extern "C" {
#endif

/// VisClassification
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visClassificationNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result, jfloat score_threshold, jfloat font_size,
    jobjectArray labels) {
  return fni::VisClassificationFromJava(env, argb8888_bitmap, result,
                                        score_threshold, font_size, labels);
}

/// VisDetection
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visDetectionNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result, jfloat score_threshold, jint line_size,
    jfloat font_size, jobjectArray labels) {
  return fni::VisDetectionFromJava(env, argb8888_bitmap, result, score_threshold,
                                   line_size, font_size, labels);
}

/// VisOcr
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visOcrNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result) {
  return fni::VisOcrFromJava(env, argb8888_bitmap, result);
}

/// VisSegmentation
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visSegmentationNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result, jfloat weight) {
  // First, try visualize segmentation result via CxxBuffer.
  if (fni::VisSegmentationFromCxxBuffer(
      env, argb8888_bitmap, result, weight)) {
    return JNI_TRUE;
  }
  // Then, try visualize segmentation from Java result(may cost some times).
  return fni::VisSegmentationFromJava(env, argb8888_bitmap, result, weight);
}

/// VisFaceDetection
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visFaceDetectionNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result, jint line_size, jfloat font_size) {
  return fni::VisFaceDetectionFromJava(env, argb8888_bitmap, result,
                                       line_size, font_size);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visKeyPointDetectionNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobject result, jfloat conf_threshold) {
  return fni::VisKeyPointDetectionFromJava(env, argb8888_bitmap, result,
                                           conf_threshold);
}

#ifdef __cplusplus
}
#endif

