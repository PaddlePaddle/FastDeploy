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
#include <jni.h>

#include "fastdeploy_jni.h"

#ifdef __cplusplus
extern "C" {
#endif

/// VisDetection
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visDetectionNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap, jobjectArray boxes,
    jfloatArray scores, jintArray label_ids, jfloat score_threshold,
    jint line_size, jfloat font_size, jobjectArray labels) {
  // Draw DetectionResult to ARGB8888 Bitmap
  int len = env->GetArrayLength(boxes);
  if ((len == 0) || (len != env->GetArrayLength(scores)) ||
      (len != env->GetArrayLength(label_ids))) {
    return JNI_FALSE;
  }
  fastdeploy::vision::DetectionResult c_result;
  c_result.Resize(len);

  // boxes [n,4]
  bool check_validation = true;
  for (int i = 0; i < len; ++i) {
    auto j_box =
        reinterpret_cast<jfloatArray>(env->GetObjectArrayElement(boxes, i));
    if (env->GetArrayLength(j_box) == 4) {
      jfloat *j_box_ptr = env->GetFloatArrayElements(j_box, nullptr);
      std::memcpy(c_result.boxes[i].data(), j_box_ptr, 4 * sizeof(float));
      env->ReleaseFloatArrayElements(j_box, j_box_ptr, 0);
    } else {
      check_validation = false;
      break;
    }
  }
  if (!check_validation) {
    return JNI_FALSE;
  }
  // scores [n]
  jfloat *j_scores_ptr = env->GetFloatArrayElements(scores, nullptr);
  std::memcpy(c_result.scores.data(), j_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(scores, j_scores_ptr, 0);
  // label_ids [n]
  jint *j_label_ids_ptr = env->GetIntArrayElements(label_ids, nullptr);
  std::memcpy(c_result.label_ids.data(), j_label_ids_ptr, len * sizeof(int));
  env->ReleaseIntArrayElements(label_ids, j_label_ids_ptr, 0);

  // Get labels from Java [n]
  std::vector<std::string> c_labels;
  int label_len = env->GetArrayLength(labels);
  if (label_len > 0) {
    for (int i = 0; i < label_len; ++i) {
      auto j_str =
          reinterpret_cast<jstring>(env->GetObjectArrayElement(labels, i));
      c_labels.push_back(fastdeploy::jni::ConvertTo<std::string>(env, j_str));
    }
  }

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

/// VisClassification
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visClassificationNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap, jfloatArray scores,
    jintArray label_ids, jfloat score_threshold, jfloat font_size,
    jobjectArray labels) {
  // Draw ClassifyResult to ARGB8888 Bitmap
  int len = env->GetArrayLength(scores);
  if ((len == 0) || (len != env->GetArrayLength(label_ids))) {
    return JNI_FALSE;
  }
  fastdeploy::vision::ClassifyResult c_result;
  c_result.scores.resize(len);
  c_result.label_ids.resize(len);
  // scores [n]
  jfloat *j_scores_ptr = env->GetFloatArrayElements(scores, nullptr);
  std::memcpy(c_result.scores.data(), j_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(scores, j_scores_ptr, 0);
  // label_ids [n]
  jint *j_label_ids_ptr = env->GetIntArrayElements(label_ids, nullptr);
  std::memcpy(c_result.label_ids.data(), j_label_ids_ptr, len * sizeof(int));
  env->ReleaseIntArrayElements(label_ids, j_label_ids_ptr, 0);

  // Get labels from Java [n]
  std::vector<std::string> c_labels;
  int label_len = env->GetArrayLength(labels);
  if (label_len > 0) {
    for (int i = 0; i < label_len; ++i) {
      auto j_str =
          reinterpret_cast<jstring>(env->GetObjectArrayElement(labels, i));
      c_labels.push_back(fastdeploy::jni::ConvertTo<std::string>(env, j_str));
    }
  }
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

/// VisOcr
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_Visualize_visOcrNative(
    JNIEnv *env, jclass clazz, jobject argb8888_bitmap,
    jobjectArray boxes, jobjectArray text, jfloatArray rec_scores,
    jfloatArray cls_scores, jintArray cls_labels) {
  // Draw OCRResult to ARGB8888 Bitmap
  int len = env->GetArrayLength(boxes);
  if ((len == 0) || (len != env->GetArrayLength(text)) ||
      (len != env->GetArrayLength(rec_scores)) ||
      (len != env->GetArrayLength(cls_scores)) ||
      (len != env->GetArrayLength(cls_labels))) {
    return JNI_FALSE;
  }
  fastdeploy::vision::OCRResult c_result;
  c_result.boxes.resize(len);
  c_result.rec_scores.resize(len);
  c_result.cls_scores.resize(len);
  c_result.cls_labels.resize(len);

  // boxes [n,8]
  bool check_validation = true;
  for (int i = 0; i < len; ++i) {
    auto j_box =
        reinterpret_cast<jintArray>(env->GetObjectArrayElement(boxes, i));
    if (env->GetArrayLength(j_box) == 8) {
      jint *j_box_ptr = env->GetIntArrayElements(j_box, nullptr);
      std::memcpy(c_result.boxes[i].data(), j_box_ptr, 8 * sizeof(int));
      env->ReleaseIntArrayElements(j_box, j_box_ptr, 0);
    } else {
      check_validation = false;
      break;
    }
  }
  if (!check_validation) {
    return JNI_FALSE;
  }

  // text [n]
  int text_len = env->GetArrayLength(text);
  if (text_len > 0) {
    for (int i = 0; i < text_len; ++i) {
      auto j_str =
          reinterpret_cast<jstring>(env->GetObjectArrayElement(text, i));
      c_result.text.push_back(fastdeploy::jni::ConvertTo<std::string>(env, j_str));
    }
  }

  // rec_scores [n]
  jfloat *j_rec_scores_ptr = env->GetFloatArrayElements(rec_scores, nullptr);
  std::memcpy(c_result.rec_scores.data(), j_rec_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(rec_scores, j_rec_scores_ptr, 0);
  // cls_scores [n]
  jfloat *j_cls_scores_ptr = env->GetFloatArrayElements(cls_scores, nullptr);
  std::memcpy(c_result.cls_scores.data(), j_cls_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(cls_scores, j_cls_scores_ptr, 0);
  // cls_labels [n]
  jint *j_cls_label_ptr = env->GetIntArrayElements(cls_labels, nullptr);
  std::memcpy(c_result.cls_labels.data(), j_cls_label_ptr, len * sizeof(int));
  env->ReleaseIntArrayElements(cls_labels, j_cls_label_ptr, 0);

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

























