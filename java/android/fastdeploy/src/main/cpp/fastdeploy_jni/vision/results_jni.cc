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

#include <jni.h>             // NOLINT
#include <android/bitmap.h>  // NOLINT
#include "fastdeploy_jni/perf_jni.h"  // NOLINT
#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT

namespace fastdeploy {
namespace jni {

/// Initialize a Java Result object from native cxx_result.
bool AllocateJavaClassifyResultFromCxx(
    JNIEnv *env, jobject j_cls_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_cls_result_obj' param
  // is a ref of Java ClassifyResult.
  // Field signatures of Java ClassifyResult:
  // (1) mScores float[]  shape (n):   [F
  // (2) mLabelIds int[]  shape (n):   [I
  // (3) mInitialized boolean:         Z
  // Docs: docs/api/vision_results/classification_result.md
  if (cxx_result == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::ClassifyResult *>(cxx_result);

  const int len = static_cast<int>(c_result_ptr->scores.size());
  if (len == 0) {
    return false;
  }

  const jclass j_cls_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/ClassifyResult");
  const jfieldID j_cls_scores_id = env->GetFieldID(
      j_cls_result_clazz, "mScores", "[F");
  const jfieldID j_cls_label_ids_id = env->GetFieldID(
      j_cls_result_clazz, "mLabelIds", "[I");
  const jfieldID j_cls_initialized_id = env->GetFieldID(
      j_cls_result_clazz, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_cls_result_obj, j_cls_result_clazz)) {
    return false;
  }

  // mScores float[]  shape (n):   [F
  const auto &scores = c_result_ptr->scores;
  jfloatArray j_cls_scores_float_arr = env->NewFloatArray(len);
  env->SetFloatArrayRegion(j_cls_scores_float_arr, 0, len, scores.data());

  // mLabelIds int[]  shape (n):   [I
  const auto &label_ids = c_result_ptr->label_ids;
  jintArray j_cls_label_ids_int_arr = env->NewIntArray(len);
  env->SetIntArrayRegion(j_cls_label_ids_int_arr, 0, len, label_ids.data());

  // Set object fields
  env->SetObjectField(j_cls_result_obj, j_cls_scores_id, j_cls_scores_float_arr);
  env->SetObjectField(j_cls_result_obj, j_cls_label_ids_id, j_cls_label_ids_int_arr);
  env->SetBooleanField(j_cls_result_obj, j_cls_initialized_id, JNI_TRUE);

  // Release local Refs
  env->DeleteLocalRef(j_cls_scores_float_arr);
  env->DeleteLocalRef(j_cls_label_ids_int_arr);
  env->DeleteLocalRef(j_cls_result_clazz);

  return true;
}

bool AllocateJavaDetectionResultFromCxx(
    JNIEnv *env, jobject j_det_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_det_result_obj' param
  // is a ref of Java DetectionResult.
  // Field signatures of Java DetectionResult:
  // (1) mBoxes float[][] shape (n,4): [[F
  // (2) mScores float[]  shape (n):   [F
  // (3) mLabelIds int[]  shape (n):   [I
  // (4) mInitialized boolean:         Z
  // Docs: docs/api/vision_results/detection_result.md
  if (cxx_result == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::DetectionResult *>(cxx_result);

  const int len = static_cast<int>(c_result_ptr->boxes.size());
  if (len == 0) {
    return false;
  }

  const jclass j_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/DetectionResult");
  const jclass j_det_float_arr_clazz = env->FindClass("[F");  // (4,)
  const jfieldID j_det_boxes_id = env->GetFieldID(
      j_det_result_clazz, "mBoxes", "[[F");
  const jfieldID j_det_scores_id = env->GetFieldID(
      j_det_result_clazz, "mScores", "[F");
  const jfieldID j_det_label_ids_id = env->GetFieldID(
      j_det_result_clazz, "mLabelIds", "[I");
  const jfieldID j_det_initialized_id = env->GetFieldID(
      j_det_result_clazz, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_det_result_obj, j_det_result_clazz)) {
    return false;
  }

  // mBoxes float[][] shape (n,4): [[F
  const auto &boxes = c_result_ptr->boxes;
  jobjectArray j_det_boxes_float_arr =
      env->NewObjectArray(len, j_det_float_arr_clazz, NULL);
  for (int i = 0; i < len; ++i) {
    jfloatArray j_box = env->NewFloatArray(4);
    env->SetFloatArrayRegion(j_box, 0, 4, boxes.at(i).data());
    env->SetObjectArrayElement(j_det_boxes_float_arr, i, j_box);
    env->DeleteLocalRef(j_box);
  }

  // mScores float[]  shape (n):   [F
  const auto &scores = c_result_ptr->scores;
  jfloatArray j_det_scores_float_arr = env->NewFloatArray(len);
  env->SetFloatArrayRegion(j_det_scores_float_arr, 0, len, scores.data());

  // mLabelIds int[]  shape (n):   [I
  const auto &label_ids = c_result_ptr->label_ids;
  jintArray j_det_label_ids_int_arr = env->NewIntArray(len);
  env->SetIntArrayRegion(j_det_label_ids_int_arr, 0, len, label_ids.data());

  // Set object fields
  env->SetObjectField(j_det_result_obj, j_det_boxes_id, j_det_boxes_float_arr);
  env->SetObjectField(j_det_result_obj, j_det_scores_id, j_det_scores_float_arr);
  env->SetObjectField(j_det_result_obj, j_det_label_ids_id, j_det_label_ids_int_arr);
  env->SetBooleanField(j_det_result_obj, j_det_initialized_id, JNI_TRUE);

  // Release local Refs
  env->DeleteLocalRef(j_det_boxes_float_arr);
  env->DeleteLocalRef(j_det_scores_float_arr);
  env->DeleteLocalRef(j_det_label_ids_int_arr);
  env->DeleteLocalRef(j_det_result_clazz);
  env->DeleteLocalRef(j_det_float_arr_clazz);

  return true;
}

bool AllocateJavaOCRResultFromCxx(
    JNIEnv *env, jobject j_ocr_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_ocr_result_obj' param is a ref of Java OCRResult.
  // Field signatures of Java OCRResult:
  // (1) mBoxes int[][] shape (n,8):      [[I
  // (2) mText String[] shape (n):        [Ljava/lang/String;
  // (3) mRecScores float[]  shape (n):   [F
  // (4) mClsScores float[]  shape (n):   [F
  // (5) mClsLabels int[]  shape (n):     [I
  // (6) mInitialized boolean:            Z
  // Docs: docs/api/vision_results/ocr_result.md
  if (cxx_result == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::OCRResult *>(cxx_result);
  const int len = static_cast<int>(c_result_ptr->boxes.size());
  if (len == 0) {
    return false;
  }

  const jclass j_ocr_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/OCRResult");
  const jclass j_ocr_int_arr_clazz = env->FindClass("[I");  // (8,)
  const jclass j_ocr_str_clazz = env->FindClass("java/lang/String");
  const jfieldID j_ocr_boxes_id = env->GetFieldID(
      j_ocr_result_clazz, "mBoxes", "[[I");
  const jfieldID j_ocr_text_id = env->GetFieldID(
      j_ocr_result_clazz, "mText", "[Ljava/lang/String;");
  const jfieldID j_ocr_rec_scores_id = env->GetFieldID(
      j_ocr_result_clazz, "mRecScores", "[F");
  const jfieldID j_ocr_cls_scores_id = env->GetFieldID(
      j_ocr_result_clazz, "mClsScores", "[F");
  const jfieldID j_ocr_cls_labels_id = env->GetFieldID(
      j_ocr_result_clazz, "mClsLabels", "[I");
  const jfieldID j_ocr_initialized_id = env->GetFieldID(
      j_ocr_result_clazz, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_ocr_result_obj, j_ocr_result_clazz)) {
    return false;
  }

  // mBoxes int[][] shape (n,8):      [[I
  const auto &boxes = c_result_ptr->boxes;
  jobjectArray j_ocr_boxes_int_arr =
      env->NewObjectArray(len, j_ocr_int_arr_clazz, NULL);
  for (int i = 0; i < len; ++i) {
    jintArray j_box = env->NewIntArray(8);
    env->SetIntArrayRegion(j_box, 0, 8, boxes.at(i).data());
    env->SetObjectArrayElement(j_ocr_boxes_int_arr, i, j_box);
    env->DeleteLocalRef(j_box);
  }

  // mText String[] shape (n):        [Ljava/lang/String;
  const auto &text = c_result_ptr->text;
  jobjectArray j_ocr_text_arr =
      env->NewObjectArray(len, j_ocr_str_clazz, env->NewStringUTF(""));
  for (int64_t i = 0; i < len; ++i) {
    env->SetObjectArrayElement(j_ocr_text_arr, i, ConvertTo<jstring>(env, text.at(i)));
  }

  // mRecScores float[]  shape (n):   [F
  const auto &rec_scores = c_result_ptr->rec_scores;
  jfloatArray j_ocr_rec_scores_float_arr = env->NewFloatArray(len);
  env->SetFloatArrayRegion(j_ocr_rec_scores_float_arr, 0, len, rec_scores.data());

  const int cls_len = static_cast<int>(c_result_ptr->cls_scores.size());
  if (cls_len > 0) {
    // mClsScores float[]  shape (n):   [F
    const auto &cls_scores = c_result_ptr->cls_scores;
    jfloatArray j_ocr_cls_scores_float_arr = env->NewFloatArray(cls_len);
    env->SetFloatArrayRegion(j_ocr_cls_scores_float_arr, 0, cls_len, cls_scores.data());

    // mClsLabels int[]  shape (n):     [I
    const auto &cls_labels = c_result_ptr->cls_labels;
    jintArray j_ocr_cls_labels_int_arr = env->NewIntArray(cls_len);
    env->SetIntArrayRegion(j_ocr_cls_labels_int_arr, 0, cls_len, cls_labels.data());

    env->SetObjectField(j_ocr_result_obj, j_ocr_cls_scores_id, j_ocr_cls_scores_float_arr);
    env->SetObjectField(j_ocr_result_obj, j_ocr_cls_labels_id, j_ocr_cls_labels_int_arr);

    env->DeleteLocalRef(j_ocr_cls_scores_float_arr);
    env->DeleteLocalRef(j_ocr_cls_labels_int_arr);
  }

  // Set object fields
  env->SetObjectField(j_ocr_result_obj, j_ocr_boxes_id, j_ocr_boxes_int_arr);
  env->SetObjectField(j_ocr_result_obj, j_ocr_text_id, j_ocr_text_arr);
  env->SetObjectField(j_ocr_result_obj, j_ocr_rec_scores_id, j_ocr_rec_scores_float_arr);
  env->SetBooleanField(j_ocr_result_obj, j_ocr_initialized_id, JNI_TRUE);

  // Release local Refs
  env->DeleteLocalRef(j_ocr_boxes_int_arr);
  env->DeleteLocalRef(j_ocr_text_arr);
  env->DeleteLocalRef(j_ocr_rec_scores_float_arr);
  env->DeleteLocalRef(j_ocr_result_clazz);
  env->DeleteLocalRef(j_ocr_int_arr_clazz);
  env->DeleteLocalRef(j_ocr_str_clazz);

  return true;
}

bool AllocateJavaSegmentationResultFromCxx(
    JNIEnv *env, jobject j_seg_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_seg_result_obj' param is
  // a ref of Java SegmentationResult.
  // Field signatures of Java SegmentationResult:
  // (1) mLabelMap int[] shape (n):        [I
  // (2) mShape long[]  shape (2) (H,W):   [J
  // (3) mContainScoreMap boolean:         Z
  // (4) mScoreMap float[]  shape (n):     [F
  // (5) mInitialized boolean:             Z
  // Docs: docs/api/vision_results/segmentation_result.md
  if (cxx_result == nullptr) {
    return false;
  }
  auto c_result_ptr =
      reinterpret_cast<vision::SegmentationResult *>(cxx_result);

  const int len = static_cast<int>(c_result_ptr->label_map.size());
  if (len == 0) {
    return false;
  }

  const jclass j_seg_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/SegmentationResult");
  const jfieldID j_seg_label_map_id = env->GetFieldID(
      j_seg_result_clazz, "mLabelMap", "[B");
  const jfieldID j_seg_shape_id = env->GetFieldID(
      j_seg_result_clazz, "mShape", "[J");
  const jfieldID j_seg_contain_shape_map_id = env->GetFieldID(
      j_seg_result_clazz, "mContainScoreMap", "Z");
  const jfieldID j_seg_score_map_id = env->GetFieldID(
      j_seg_result_clazz, "mScoreMap", "[F");
  const jfieldID j_enable_cxx_buffer_id = env->GetFieldID(
      j_seg_result_clazz, "mEnableCxxBuffer", "Z");
  const jfieldID  j_cxx_buffer_id = env->GetFieldID(
      j_seg_result_clazz, "mCxxBuffer", "J");
  const jfieldID j_seg_initialized_id = env->GetFieldID(
      j_seg_result_clazz, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_seg_result_obj, j_seg_result_clazz)) {
    return false;
  }

  // If 'mEnableCxxBuffer' set as true, then, we only setup the cxx result
  // pointer to the value of 'mCxxBuffer' field. Some users may want
  // to use this method to boost the performance of segmentation.
  jboolean j_enable_cxx_buffer =
      env->GetBooleanField(j_seg_result_obj, j_enable_cxx_buffer_id);
  if (j_enable_cxx_buffer == JNI_TRUE) {
    jlong j_cxx_buffer = reinterpret_cast<jlong>(c_result_ptr);
    env->SetLongField(j_seg_result_obj, j_cxx_buffer_id, j_cxx_buffer);
    env->SetBooleanField(j_seg_result_obj, j_seg_initialized_id, JNI_TRUE);
    return true;
  }

  // mLabelMap int[] shape (n):        [I
  const auto &label_map_uint8 = c_result_ptr->label_map;
  jbyteArray j_seg_label_map_byte_arr = env->NewByteArray(len);
  env->SetByteArrayRegion(j_seg_label_map_byte_arr, 0, len,
                          reinterpret_cast<jbyte*>(const_cast<uint8_t*>(
                              label_map_uint8.data())));

  // mShape long[]  shape (2) (H,W):   [J
  const auto &shape = c_result_ptr->shape;
  const int shape_len = static_cast<int>(shape.size());
  jlongArray j_seg_shape_long_arr = env->NewLongArray(shape_len);
  env->SetLongArrayRegion(j_seg_shape_long_arr, 0, shape_len, shape.data());

  // mContainScoreMap boolean:         Z
  const auto &contain_score_map = c_result_ptr->contain_score_map;
  if (contain_score_map) {
    env->SetBooleanField(j_seg_result_obj, j_seg_contain_shape_map_id, JNI_TRUE);
  }

  // mScoreMap float[]  shape (n):     [F
  if (contain_score_map) {
    const auto &score_map = c_result_ptr->score_map;
    jfloatArray j_seg_score_map_float_arr = env->NewFloatArray(len);
    env->SetFloatArrayRegion(j_seg_score_map_float_arr, 0, len, score_map.data());
    env->SetObjectField(j_seg_result_obj, j_seg_score_map_id, j_seg_score_map_float_arr);
    env->DeleteLocalRef(j_seg_score_map_float_arr);
  }

  // Set object fields
  env->SetObjectField(j_seg_result_obj, j_seg_label_map_id, j_seg_label_map_byte_arr);
  env->SetObjectField(j_seg_result_obj, j_seg_shape_id, j_seg_shape_long_arr);
  env->SetBooleanField(j_seg_result_obj, j_seg_initialized_id, JNI_TRUE);

  // Release local Refs
  // env->DeleteLocalRef(j_seg_label_map_int_arr);
  env->DeleteLocalRef(j_seg_label_map_byte_arr);
  env->DeleteLocalRef(j_seg_shape_long_arr);
  env->DeleteLocalRef(j_seg_result_clazz);

  return true;
}

bool AllocateJavaFaceDetectionResultFromCxx(
    JNIEnv *env, jobject j_face_det_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_face_det_result_obj' param
  // is a ref of Java FaceDetectionResult.
  // Field signatures of Java FaceDetectionResult:
  // (1) mBoxes float[][] shape (n,4):     [[F
  // (2) mScores float[]  shape (n):       [F
  // (3) mLandmarks float[][] shape (n,2): [[F
  // (4) mLandmarksPerFace int:            I
  // (5) mInitialized boolean:             Z
  // Docs: docs/api/vision_results/face_detection_result.md
  if (cxx_result == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::FaceDetectionResult *>(cxx_result);

  const int len = static_cast<int>(c_result_ptr->boxes.size());
  if (len == 0) {
    return false;
  }

  const jclass j_face_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/FaceDetectionResult");
  const jclass j_face_det_float_arr_clazz = env->FindClass("[F");  // (4|2,)
  const jfieldID j_face_det_boxes_id = env->GetFieldID(
      j_face_det_result_clazz, "mBoxes", "[[F");
  const jfieldID j_face_det_scores_id = env->GetFieldID(
      j_face_det_result_clazz, "mScores", "[F");
  const jfieldID j_face_det_landmarks_id = env->GetFieldID(
      j_face_det_result_clazz, "mLandmarks", "[[F");
  const jfieldID j_face_det_landmarks_per_face_id = env->GetFieldID(
      j_face_det_result_clazz, "mLandmarksPerFace", "I");
  const jfieldID j_face_det_initialized_id = env->GetFieldID(
      j_face_det_result_clazz, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_face_det_result_obj, j_face_det_result_clazz)) {
    return false;
  }

  // mBoxes float[][] shape (n,4):      [[F
  const auto &boxes = c_result_ptr->boxes;
  jobjectArray j_face_det_boxes_float_arr = env->NewObjectArray(
      len, j_face_det_float_arr_clazz, NULL);
  for (int i = 0; i < len; ++i) {
    jfloatArray j_box = env->NewFloatArray(4);
    env->SetFloatArrayRegion(j_box, 0, 4, boxes.at(i).data());
    env->SetObjectArrayElement(j_face_det_boxes_float_arr, i, j_box);
    env->DeleteLocalRef(j_box);
  }

  // mScores float[]  shape (n):       [F
  const auto &scores = c_result_ptr->scores;
  jfloatArray j_face_det_scores_float_arr = env->NewFloatArray(len);
  env->SetFloatArrayRegion(j_face_det_scores_float_arr, 0, len, scores.data());

  // mLandmarksPerFace int:            I
  const jint landmarks_per_face = static_cast<jint>(c_result_ptr->landmarks_per_face);
  env->SetIntField(j_face_det_result_obj, j_face_det_landmarks_per_face_id, landmarks_per_face);

  // mLandmarks float[][] shape (n,2): [[F
  if (landmarks_per_face > 0) {
    const auto &landmarks = c_result_ptr->landmarks;
    const int landmarks_len = static_cast<int>(landmarks.size());
    jobjectArray j_face_det_landmarks_float_arr = env->NewObjectArray(
        landmarks_len, j_face_det_float_arr_clazz, NULL);
    for (int i = 0; i < landmarks_len; ++i) {
      jfloatArray j_landmark = env->NewFloatArray(2);
      env->SetFloatArrayRegion(j_landmark, 0, 2, landmarks.at(i).data());
      env->SetObjectArrayElement(j_face_det_landmarks_float_arr, i, j_landmark);
      env->DeleteLocalRef(j_landmark);
    }
    env->SetObjectField(j_face_det_result_obj, j_face_det_landmarks_id,
                        j_face_det_landmarks_float_arr);
    env->DeleteLocalRef(j_face_det_landmarks_float_arr);
  }

  // Set object fields
  env->SetObjectField(j_face_det_result_obj, j_face_det_boxes_id, j_face_det_boxes_float_arr);
  env->SetObjectField(j_face_det_result_obj, j_face_det_scores_id, j_face_det_scores_float_arr);
  env->SetBooleanField(j_face_det_result_obj, j_face_det_initialized_id, JNI_TRUE);

  // Release local Refs
  env->DeleteLocalRef(j_face_det_boxes_float_arr);
  env->DeleteLocalRef(j_face_det_scores_float_arr);
  env->DeleteLocalRef(j_face_det_result_clazz);
  env->DeleteLocalRef(j_face_det_float_arr_clazz);

  return true;
}

bool AllocateJavaKeyPointDetectionResultFromCxx(
    JNIEnv *env, jobject j_keypoint_det_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_keypoint_det_result_obj' param
  // is a ref of Java KeyPointDetectionResult.
  // Field signatures of Java KeyPointDetectionResult:
  // (1) mBoxes float[][] shape (n*num_joints,2): [[F
  // (2) mScores float[]  shape (n*num_joints):   [F
  // (3) mNumJoints int  shape (1):               I
  // (4) mInitialized boolean:                    Z
  // Docs: docs/api/vision_results/keypointdetection_result.md
  if (cxx_result == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::KeyPointDetectionResult *>(cxx_result);

  const int len = static_cast<int>(c_result_ptr->keypoints.size());
  if (len == 0) {
    return false;
  }

  const jclass j_keypoint_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/KeyPointDetectionResult");
  const jclass j_keypoint_float_arr_clazz = env->FindClass("[F");  // (2,)
  const jfieldID j_keypoint_det_keypoints_id = env->GetFieldID(
      j_keypoint_det_result_clazz, "mKeyPoints", "[[F");
  const jfieldID j_keypoint_det_scores_id = env->GetFieldID(
      j_keypoint_det_result_clazz, "mScores", "[F");
  const jfieldID j_keypoint_det_num_joints_id = env->GetFieldID(
      j_keypoint_det_result_clazz, "mNumJoints", "I");
  const jfieldID j_keypoint_det_initialized_id = env->GetFieldID(
      j_keypoint_det_result_clazz, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_keypoint_det_result_obj, j_keypoint_det_result_clazz)) {
    return false;
  }

  // mKeyPoints float[][] shape (n*num_joints,2): [[F
  const auto &keypoints = c_result_ptr->keypoints;
  jobjectArray j_keypoint_det_keypoints_float_arr =
      env->NewObjectArray(len, j_keypoint_float_arr_clazz, NULL);
  for (int i = 0; i < len; ++i) {
    jfloatArray j_point = env->NewFloatArray(2);
    env->SetFloatArrayRegion(j_point, 0, 2, keypoints.at(i).data());
    env->SetObjectArrayElement(j_keypoint_det_keypoints_float_arr, i, j_point);
    env->DeleteLocalRef(j_point);
  }

  // mScores float[]  shape (n):   [F
  const auto &scores = c_result_ptr->scores;
  const int score_len = scores.size();
  jfloatArray j_keypoint_det_scores_float_arr = env->NewFloatArray(score_len);
  env->SetFloatArrayRegion(j_keypoint_det_scores_float_arr, 0, score_len, scores.data());

  // mNumJoints int  shape (1):   I
  jint j_keypoint_det_num_joints = static_cast<jint>(c_result_ptr->num_joints);

  // Set object fields
  env->SetObjectField(j_keypoint_det_result_obj, j_keypoint_det_keypoints_id, j_keypoint_det_keypoints_float_arr);
  env->SetObjectField(j_keypoint_det_result_obj, j_keypoint_det_scores_id, j_keypoint_det_scores_float_arr);
  env->SetIntField(j_keypoint_det_result_obj, j_keypoint_det_num_joints_id, j_keypoint_det_num_joints);
  env->SetBooleanField(j_keypoint_det_result_obj, j_keypoint_det_initialized_id, JNI_TRUE);

  // Release local Refs
  env->DeleteLocalRef(j_keypoint_det_keypoints_float_arr);
  env->DeleteLocalRef(j_keypoint_det_scores_float_arr);
  env->DeleteLocalRef(j_keypoint_det_result_clazz);
  env->DeleteLocalRef(j_keypoint_float_arr_clazz);

  return true;
}

bool AllocateJavaResultFromCxx(JNIEnv *env, jobject j_result_obj,
                               void *cxx_result, vision::ResultType type) {
  if (type == vision::ResultType::CLASSIFY) {
    return AllocateJavaClassifyResultFromCxx(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::DETECTION) {
    return AllocateJavaDetectionResultFromCxx(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::OCR) {
    return AllocateJavaOCRResultFromCxx(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::SEGMENTATION) {
    return AllocateJavaSegmentationResultFromCxx(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::FACE_DETECTION) {
    return AllocateJavaFaceDetectionResultFromCxx(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::KEYPOINT_DETECTION) {
    return AllocateJavaKeyPointDetectionResultFromCxx(env, j_result_obj, cxx_result);
  } else {
    LOGE("Not support this ResultType in JNI now, type: %d",
         static_cast<int>(type));
    return false;
  }
}

/// New a Java Result object from native result cxx_result.
jobject NewJavaClassifyResultFromCxx(JNIEnv *env, void *cxx_result) {
  const jclass j_cls_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/ClassifyResult");
  const jmethodID j_cls_result_init = env->GetMethodID(
      j_cls_result_clazz, "<init>", "()V");
  jobject j_cls_result_obj = env->NewObject(
      j_cls_result_clazz, j_cls_result_init);
  AllocateJavaClassifyResultFromCxx(env, j_cls_result_obj, cxx_result);
  env->DeleteLocalRef(j_cls_result_clazz);
  return j_cls_result_obj;
}

jobject NewJavaDetectionResultFromCxx(JNIEnv *env, void *cxx_result) {
  const jclass j_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/DetectionResult");
  const jmethodID j_det_result_init = env->GetMethodID(
      j_det_result_clazz, "<init>", "()V");
  jobject j_det_result_obj = env->NewObject(
      j_det_result_clazz, j_det_result_init);
  AllocateJavaDetectionResultFromCxx(env, j_det_result_obj, cxx_result);
  env->DeleteLocalRef(j_det_result_clazz);
  return j_det_result_obj;
}

jobject NewJavaOCRResultFromCxx(JNIEnv *env, void *cxx_result) {
  const jclass j_ocr_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/OCRResult");
  const jmethodID j_ocr_result_init = env->GetMethodID(
      j_ocr_result_clazz, "<init>", "()V");
  jobject j_ocr_result_obj = env->NewObject(
      j_ocr_result_clazz, j_ocr_result_init);
  AllocateJavaOCRResultFromCxx(env, j_ocr_result_obj, cxx_result);
  env->DeleteLocalRef(j_ocr_result_clazz);
  return j_ocr_result_obj;
}

jobject NewJavaSegmentationResultFromCxx(JNIEnv *env, void *cxx_result) {
  const jclass j_seg_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/SegmentationResult");
  const jmethodID j_seg_result_init = env->GetMethodID(
      j_seg_result_clazz, "<init>", "()V");
  jobject j_seg_result_obj = env->NewObject(
      j_seg_result_clazz, j_seg_result_init);
  AllocateJavaSegmentationResultFromCxx(env, j_seg_result_obj, cxx_result);
  env->DeleteLocalRef(j_seg_result_clazz);
  return j_seg_result_obj;
}

jobject NewJavaFaceDetectionResultFromCxx(JNIEnv *env, void *cxx_result) {
  const jclass j_face_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/FaceDetectionResult");
  const jmethodID j_face_det_result_init = env->GetMethodID(
      j_face_det_result_clazz, "<init>", "()V");
  jobject j_face_det_result_obj = env->NewObject(
      j_face_det_result_clazz, j_face_det_result_init);
  AllocateJavaFaceDetectionResultFromCxx(env, j_face_det_result_obj, cxx_result);
  env->DeleteLocalRef(j_face_det_result_clazz);
  return j_face_det_result_obj;
}

jobject NewJavaKeyPointDetectionResultFromCxx(JNIEnv *env, void *cxx_result) {
  const jclass j_keypoint_det_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/KeyPointDetectionResult");
  const jmethodID j_keypoint_det_result_init = env->GetMethodID(
      j_keypoint_det_result_clazz, "<init>", "()V");
  jobject j_keypoint_det_result_obj = env->NewObject(
      j_keypoint_det_result_clazz, j_keypoint_det_result_init);
  AllocateJavaKeyPointDetectionResultFromCxx(env, j_keypoint_det_result_obj, cxx_result);
  env->DeleteLocalRef(j_keypoint_det_result_clazz);
  return j_keypoint_det_result_obj;
}

jobject NewJavaResultFromCxx(
    JNIEnv *env, void *cxx_result, vision::ResultType type) {
  if (type == vision::ResultType::CLASSIFY) {
    return NewJavaClassifyResultFromCxx(env, cxx_result);
  } else if (type == vision::ResultType::DETECTION) {
    return NewJavaDetectionResultFromCxx(env, cxx_result);
  } else if (type == vision::ResultType::OCR) {
    return NewJavaOCRResultFromCxx(env, cxx_result);
  } else if (type == vision::ResultType::SEGMENTATION) {
    return NewJavaSegmentationResultFromCxx(env, cxx_result);
  } else if (type == vision::ResultType::FACE_DETECTION) {
    return NewJavaFaceDetectionResultFromCxx(env, cxx_result);
  } else if (type == vision::ResultType::KEYPOINT_DETECTION) {
    return NewJavaKeyPointDetectionResultFromCxx(env, cxx_result);
  } else {
    LOGE("Not support this ResultType in JNI now, type: %d",
         static_cast<int>(type));
    return NULL;
  }
}

/// Init Cxx result from Java Result
bool AllocateClassifyResultFromJava(
    JNIEnv *env, jobject j_cls_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_cls_result_obj' param
  // is a ref of Java ClassifyResult.
  // Field signatures of Java ClassifyResult:
  // (1) mScores float[]  shape (n):   [F
  // (2) mLabelIds int[]  shape (n):   [I
  // (3) mInitialized boolean:         Z
  // Docs: docs/api/vision_results/classification_result.md
  if (cxx_result == nullptr || j_cls_result_obj == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::ClassifyResult *>(cxx_result);

  const jclass j_cls_result_clazz_cc = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/ClassifyResult");
  const jfieldID j_cls_scores_id_cc = env->GetFieldID(
      j_cls_result_clazz_cc, "mScores", "[F");
  const jfieldID j_cls_label_ids_id_cc = env->GetFieldID(
      j_cls_result_clazz_cc, "mLabelIds", "[I");
  const jfieldID j_cls_initialized_id_cc = env->GetFieldID(
      j_cls_result_clazz_cc, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_cls_result_obj, j_cls_result_clazz_cc)) {
    return false;
  }

  // mInitialized boolean:         Z
  jboolean j_cls_initialized =
      env->GetBooleanField(j_cls_result_obj, j_cls_initialized_id_cc);
  if (j_cls_initialized == JNI_FALSE) {
    return false;
  }

  jfloatArray j_cls_scores_float_arr = reinterpret_cast<jfloatArray>(
      env->GetObjectField(j_cls_result_obj, j_cls_scores_id_cc));
  jintArray j_cls_label_ids_int_arr = reinterpret_cast<jintArray>(
      env->GetObjectField(j_cls_result_obj, j_cls_label_ids_id_cc));

  const int len = env->GetArrayLength(j_cls_scores_float_arr);
  if ((len == 0) || (len != env->GetArrayLength(j_cls_label_ids_int_arr))) {
    return false;
  }

  // Init Cxx result
  c_result_ptr->Clear();
  c_result_ptr->scores.resize(len);
  c_result_ptr->label_ids.resize(len);

  // mScores float[]  shape (n):   [F
  jfloat *j_cls_scores_ptr =
      env->GetFloatArrayElements(j_cls_scores_float_arr, nullptr);
  std::memcpy(c_result_ptr->scores.data(), j_cls_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(j_cls_scores_float_arr, j_cls_scores_ptr, 0);

  // mLabelIds int[]  shape (n):   [I
  jint *j_cls_label_ids_ptr =
      env->GetIntArrayElements(j_cls_label_ids_int_arr, nullptr);
  std::memcpy(c_result_ptr->label_ids.data(), j_cls_label_ids_ptr, len * sizeof(int));
  env->ReleaseIntArrayElements(j_cls_label_ids_int_arr, j_cls_label_ids_ptr, 0);

  // Release local Refs
  env->DeleteLocalRef(j_cls_result_clazz_cc);

  return true;
}

bool AllocateDetectionResultFromJava(
    JNIEnv *env, jobject j_det_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_det_result_obj' param
  // is a ref of Java DetectionResult.
  // Field signatures of Java DetectionResult:
  // (1) mBoxes float[][] shape (n,4): [[F
  // (2) mScores float[]  shape (n):   [F
  // (3) mLabelIds int[]  shape (n):   [I
  // (4) mInitialized boolean:         Z
  // Docs: docs/api/vision_results/detection_result.md
  if (cxx_result == nullptr || j_det_result_obj == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::DetectionResult *>(cxx_result);

  const jclass j_det_result_clazz_cc = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/DetectionResult");
  const jfieldID j_det_boxes_id_cc = env->GetFieldID(
      j_det_result_clazz_cc, "mBoxes", "[[F");
  const jfieldID j_det_scores_id_cc = env->GetFieldID(
      j_det_result_clazz_cc, "mScores", "[F");
  const jfieldID j_det_label_ids_id_cc = env->GetFieldID(
      j_det_result_clazz_cc, "mLabelIds", "[I");
  const jfieldID j_det_initialized_id_cc = env->GetFieldID(
      j_det_result_clazz_cc, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_det_result_obj, j_det_result_clazz_cc)) {
    return false;
  }

  // mInitialized boolean:         Z
  jboolean j_det_initialized =
      env->GetBooleanField(j_det_result_obj, j_det_initialized_id_cc);
  if (j_det_initialized == JNI_FALSE) {
    return false;
  }

  jobjectArray j_det_boxes_float_arr = reinterpret_cast<jobjectArray>(
      env->GetObjectField(j_det_result_obj, j_det_boxes_id_cc));
  jfloatArray j_det_scores_float_arr = reinterpret_cast<jfloatArray>(
      env->GetObjectField(j_det_result_obj, j_det_scores_id_cc));
  jintArray j_det_label_ids_int_arr = reinterpret_cast<jintArray>(
      env->GetObjectField(j_det_result_obj, j_det_label_ids_id_cc));

  int len = env->GetArrayLength(j_det_boxes_float_arr);
  if ((len == 0) || (len != env->GetArrayLength(j_det_scores_float_arr)) ||
      (len != env->GetArrayLength(j_det_label_ids_int_arr))) {
    return false;
  }

  // Init Cxx result
  c_result_ptr->Clear();
  c_result_ptr->Resize(len);

  // mBoxes float[][] shape (n,4): [[F
  bool c_check_validation = true;
  for (int i = 0; i < len; ++i) {
    auto j_box = reinterpret_cast<jfloatArray>(
        env->GetObjectArrayElement(j_det_boxes_float_arr, i));
    if (env->GetArrayLength(j_box) == 4) {
      jfloat *j_box_ptr = env->GetFloatArrayElements(j_box, nullptr);
      std::memcpy(c_result_ptr->boxes[i].data(), j_box_ptr, 4 * sizeof(float));
      env->ReleaseFloatArrayElements(j_box, j_box_ptr, 0);
    } else {
      c_check_validation = false;
      break;
    }
  }
  if (!c_check_validation) {
    LOGE("The length of each detection box is not equal 4!");
    return false;
  }

  // mScores float[]  shape (n):   [F
  jfloat *j_det_scores_ptr =
      env->GetFloatArrayElements(j_det_scores_float_arr, nullptr);
  std::memcpy(c_result_ptr->scores.data(), j_det_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(j_det_scores_float_arr, j_det_scores_ptr, 0);

  // mLabelIds int[]  shape (n):   [I
  jint *j_det_label_ids_ptr =
      env->GetIntArrayElements(j_det_label_ids_int_arr, nullptr);
  std::memcpy(c_result_ptr->label_ids.data(), j_det_label_ids_ptr, len * sizeof(int));
  env->ReleaseIntArrayElements(j_det_label_ids_int_arr, j_det_label_ids_ptr, 0);

  // Release local Refs
  env->DeleteLocalRef(j_det_result_clazz_cc);

  return true;
}

bool AllocateOCRResultFromJava(
    JNIEnv *env, jobject j_ocr_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_ocr_result_obj' param is a ref of
  // Java OCRResult. Field signatures of Java OCRResult:
  // (1) mBoxes int[][] shape (n,8):      [[I
  // (2) mText String[] shape (n):        [Ljava/lang/String;
  // (3) mRecScores float[]  shape (n):   [F
  // (4) mClsScores float[]  shape (n):   [F
  // (5) mClsLabels int[]  shape (n):     [I
  // (6) mInitialized boolean:            Z
  // Docs: docs/api/vision_results/ocr_result.md
  if (cxx_result == nullptr || j_ocr_result_obj == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::OCRResult *>(cxx_result);

  const jclass j_ocr_result_clazz_cc = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/OCRResult");
  const jfieldID j_ocr_boxes_id_cc = env->GetFieldID(
      j_ocr_result_clazz_cc, "mBoxes", "[[I");
  const jfieldID j_ocr_text_id_cc = env->GetFieldID(
      j_ocr_result_clazz_cc, "mText", "[Ljava/lang/String;");
  const jfieldID j_ocr_rec_scores_id_cc = env->GetFieldID(
      j_ocr_result_clazz_cc, "mRecScores", "[F");
  const jfieldID j_ocr_cls_scores_id_cc = env->GetFieldID(
      j_ocr_result_clazz_cc, "mClsScores", "[F");
  const jfieldID j_ocr_cls_labels_id_cc = env->GetFieldID(
      j_ocr_result_clazz_cc, "mClsLabels", "[I");
  const jfieldID j_ocr_initialized_id_cc = env->GetFieldID(
      j_ocr_result_clazz_cc, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_ocr_result_obj, j_ocr_result_clazz_cc)) {
    return false;
  }

  // mInitialized boolean:         Z
  jboolean j_ocr_initialized =
      env->GetBooleanField(j_ocr_result_obj, j_ocr_initialized_id_cc);
  if (j_ocr_initialized == JNI_FALSE) {
    return false;
  }

  jobjectArray j_ocr_boxes_arr = reinterpret_cast<jobjectArray>(
      env->GetObjectField(j_ocr_result_obj, j_ocr_boxes_id_cc));
  jobjectArray j_ocr_text_arr = reinterpret_cast<jobjectArray>(
      env->GetObjectField(j_ocr_result_obj, j_ocr_text_id_cc));
  jfloatArray j_ocr_rec_scores_float_arr = reinterpret_cast<jfloatArray>(
      env->GetObjectField(j_ocr_result_obj, j_ocr_rec_scores_id_cc));
  jfloatArray j_ocr_cls_scores_float_arr = reinterpret_cast<jfloatArray>(
      env->GetObjectField(j_ocr_result_obj, j_ocr_cls_scores_id_cc));
  jintArray j_ocr_cls_labels_int_arr = reinterpret_cast<jintArray>(
      env->GetObjectField(j_ocr_result_obj, j_ocr_cls_labels_id_cc));

  const int len = env->GetArrayLength(j_ocr_boxes_arr);
  if ((len == 0) || (len != env->GetArrayLength(j_ocr_text_arr)) ||
      (len != env->GetArrayLength(j_ocr_rec_scores_float_arr))){
    return false;
  }

  int cls_len = 0;
  if ((j_ocr_cls_labels_int_arr != NULL) && (j_ocr_cls_scores_float_arr != NULL)) {
    cls_len = env->GetArrayLength(j_ocr_cls_scores_float_arr);
    if (cls_len != env->GetArrayLength(j_ocr_cls_labels_int_arr)) {
      return false;
    }
  }

  // Init cxx result
  c_result_ptr->Clear();
  c_result_ptr->boxes.resize(len);
  c_result_ptr->rec_scores.resize(len);

  if (cls_len > 0) {
    c_result_ptr->cls_scores.resize(cls_len);
    c_result_ptr->cls_labels.resize(cls_len);
  }

  // mBoxes int[][] shape (n,8):      [[I
  bool c_check_validation = true;
  for (int i = 0; i < len; ++i) {
    auto j_box = reinterpret_cast<jintArray>(
        env->GetObjectArrayElement(j_ocr_boxes_arr, i));
    if (env->GetArrayLength(j_box) == 8) {
      jint *j_box_ptr = env->GetIntArrayElements(j_box, nullptr);
      std::memcpy(c_result_ptr->boxes[i].data(), j_box_ptr, 8 * sizeof(int));
      env->ReleaseIntArrayElements(j_box, j_box_ptr, 0);
    } else {
      c_check_validation = false;
      break;
    }
  }
  if (!c_check_validation) {
    return false;
  }

  // mText String[] shape (n):        [Ljava/lang/String;
  for (int i = 0; i < len; ++i) {
    auto j_text = reinterpret_cast<jstring>(
        env->GetObjectArrayElement(j_ocr_text_arr, i));
    c_result_ptr->text.push_back(ConvertTo<std::string>(env, j_text));
  }

  // mRecScores float[]  shape (n):   [F
  jfloat *j_ocr_rec_scores_ptr =
      env->GetFloatArrayElements(j_ocr_rec_scores_float_arr, nullptr);
  std::memcpy(c_result_ptr->rec_scores.data(), j_ocr_rec_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(j_ocr_rec_scores_float_arr, j_ocr_rec_scores_ptr, 0);

  if (cls_len > 0) {
    // mClsScores float[]  shape (n):   [F
    jfloat *j_ocr_cls_scores_ptr =
        env->GetFloatArrayElements(j_ocr_cls_scores_float_arr, nullptr);
    std::memcpy(c_result_ptr->cls_scores.data(), j_ocr_cls_scores_ptr, cls_len * sizeof(float));
    env->ReleaseFloatArrayElements(j_ocr_cls_scores_float_arr, j_ocr_cls_scores_ptr, 0);

    //  mClsLabels int[]  shape (n):     [I
    jint *j_ocr_cls_labels_ptr =
        env->GetIntArrayElements(j_ocr_cls_labels_int_arr, nullptr);
    std::memcpy(c_result_ptr->cls_labels.data(), j_ocr_cls_labels_ptr, cls_len * sizeof(int));
    env->ReleaseIntArrayElements(j_ocr_cls_labels_int_arr, j_ocr_cls_labels_ptr,0);
  }

  // Release local Refs
  env->DeleteLocalRef(j_ocr_result_clazz_cc);

  return true;
}

bool AllocateSegmentationResultFromJava(
    JNIEnv *env, jobject j_seg_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_seg_result_obj' param is a ref of Java
  // SegmentationResult. Field signatures of Java SegmentationResult:
  // (1) mLabelMap int[] shape (n):        [I
  // (2) mShape long[]  shape (2) (H,W):   [J
  // (3) mContainScoreMap boolean:         Z
  // (4) mScoreMap float[]  shape (n):     [F
  // (5) mInitialized boolean:             Z
  // Docs: docs/api/vision_results/segmentation_result.md
  if (cxx_result == nullptr || j_seg_result_obj == nullptr) {
    return false;
  }
  auto c_result_ptr =
      reinterpret_cast<vision::SegmentationResult *>(cxx_result);

  const jclass j_seg_result_clazz_cc = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/SegmentationResult");
  const jfieldID j_seg_label_map_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mLabelMap", "[B");
  const jfieldID j_seg_shape_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mShape", "[J");
  const jfieldID j_seg_contain_shape_map_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mContainScoreMap", "Z");
  const jfieldID j_seg_score_map_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mScoreMap", "[F");
  const jfieldID j_enable_cxx_buffer_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mEnableCxxBuffer", "Z");
  const jfieldID  j_cxx_buffer_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mCxxBuffer", "J");
  const jfieldID j_seg_initialized_id_cc = env->GetFieldID(
      j_seg_result_clazz_cc, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_seg_result_obj, j_seg_result_clazz_cc)) {
    return false;
  }

  // mInitialized boolean:         Z
  jboolean j_seg_initialized =
      env->GetBooleanField(j_seg_result_obj, j_seg_initialized_id_cc);
  if (j_seg_initialized == JNI_FALSE) {
    return false;
  }

  // If 'mEnableCxxBuffer' set as true, then, we only Allocate from
  // cxx context to cxx result. Some users may want to use this
  // method to boost the performance of segmentation.
  jboolean j_enable_cxx_buffer =
      env->GetBooleanField(j_seg_result_obj, j_enable_cxx_buffer_id_cc);

  if (j_enable_cxx_buffer == JNI_TRUE) {
    jlong j_cxx_buffer = env->GetLongField(j_seg_result_obj, j_cxx_buffer_id_cc);
    if (j_cxx_buffer == 0) {
      return false;
    }
    // Allocate from cxx context to cxx result
    auto c_cxx_buffer = reinterpret_cast<vision::SegmentationResult *>(j_cxx_buffer);

    // (*c_result_ptr) = std::move(*c_cxx_buffer);
    c_result_ptr->shape = c_cxx_buffer->shape;
    const size_t label_len = c_cxx_buffer->label_map.size();
    c_result_ptr->label_map.resize(label_len);
    std::memcpy(c_result_ptr->label_map.data(), c_cxx_buffer->label_map.data(),
                label_len * sizeof(uint8_t));
    c_result_ptr->contain_score_map = c_cxx_buffer->contain_score_map;
    if (c_cxx_buffer->contain_score_map) {
      const size_t score_len = c_cxx_buffer->score_map.size();
      c_result_ptr->score_map.resize(score_len);
      std::memcpy(c_result_ptr->score_map.data(), c_cxx_buffer->score_map.data(),
                  score_len * sizeof(float));
    }
    return true;
  }

  jbyteArray j_seg_label_map_byte_arr = reinterpret_cast<jbyteArray>(
      env->GetObjectField(j_seg_result_obj, j_seg_label_map_id_cc));
  jlongArray j_seg_shape_long_arr = reinterpret_cast<jlongArray>(
      env->GetObjectField(j_seg_result_obj, j_seg_shape_id_cc));
  jboolean j_seg_contain_score_map =
      env->GetBooleanField(j_seg_result_obj, j_seg_contain_shape_map_id_cc);

  // Init cxx result
  c_result_ptr->Clear();
  const int label_len = env->GetArrayLength(j_seg_label_map_byte_arr);  // HxW
  const int shape_len = env->GetArrayLength(j_seg_shape_long_arr);     // 2
  c_result_ptr->label_map.resize(label_len);
  c_result_ptr->shape.resize(shape_len);

  // mLabelMap int[] shape (n):        [I
  jbyte *j_seg_label_map_byte_ptr =
      env->GetByteArrayElements(j_seg_label_map_byte_arr, nullptr);
  std::memcpy(c_result_ptr->label_map.data(), j_seg_label_map_byte_ptr, label_len * sizeof(jbyte));
  env->ReleaseByteArrayElements(j_seg_label_map_byte_arr, j_seg_label_map_byte_ptr,0);

  // mShape long[]  shape (2) (H,W):   [J
  jlong *j_seg_shape_long_ptr =
      env->GetLongArrayElements(j_seg_shape_long_arr, nullptr);
  std::memcpy(c_result_ptr->shape.data(), j_seg_shape_long_ptr, shape_len * sizeof(int64_t));
  env->ReleaseLongArrayElements(j_seg_shape_long_arr, j_seg_shape_long_ptr, 0);

  //  mScoreMap float[]  shape (n):     [F
  if (j_seg_contain_score_map) {
    jfloatArray j_seg_score_map_float_arr = reinterpret_cast<jfloatArray>(
        env->GetObjectField(j_seg_result_obj, j_seg_score_map_id_cc));

    if (j_seg_score_map_float_arr != NULL) {
      const int score_len = env->GetArrayLength(j_seg_score_map_float_arr);  // 0 | HxW

      c_result_ptr->contain_score_map = true;
      c_result_ptr->score_map.resize(score_len);
      jfloat *j_seg_score_map_float_ptr =
          env->GetFloatArrayElements(j_seg_score_map_float_arr, nullptr);
      std::memcpy(c_result_ptr->score_map.data(), j_seg_score_map_float_ptr, score_len * sizeof(float));
      env->ReleaseFloatArrayElements(j_seg_score_map_float_arr, j_seg_score_map_float_ptr, 0);
    }
  }

  // Release local Refs
  env->DeleteLocalRef(j_seg_result_clazz_cc);

  return true;
}

bool AllocateFaceDetectionResultFromJava(
    JNIEnv *env, jobject j_face_det_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_face_det_result_obj' param is a ref of Java
  // FaceDetectionResult. Field signatures of Java FaceDetectionResult:
  // (1) mBoxes float[][] shape (n,4):     [[F
  // (2) mScores float[]  shape (n):       [F
  // (3) mLandmarks float[][] shape (n,2): [[F
  // (4) mLandmarksPerFace int:            I
  // (5) mInitialized boolean:             Z
  // Docs: docs/api/vision_results/face_detection_result.md
  if (cxx_result == nullptr || j_face_det_result_obj == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::FaceDetectionResult *>(cxx_result);

  const jclass j_face_det_result_clazz_cc = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/FaceDetectionResult");
  const jfieldID j_face_det_boxes_id_cc = env->GetFieldID(
      j_face_det_result_clazz_cc, "mBoxes", "[[F");
  const jfieldID j_face_det_scores_id_cc = env->GetFieldID(
      j_face_det_result_clazz_cc, "mScores", "[F");
  const jfieldID j_face_det_landmarks_id_cc = env->GetFieldID(
      j_face_det_result_clazz_cc, "mLandmarks", "[[F");
  const jfieldID j_face_det_landmarks_per_face_id_cc = env->GetFieldID(
      j_face_det_result_clazz_cc, "mLandmarksPerFace", "I");
  const jfieldID j_face_det_initialized_id_cc = env->GetFieldID(
      j_face_det_result_clazz_cc, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_face_det_result_obj, j_face_det_result_clazz_cc)) {
    return false;
  }

  // mInitialized boolean:         Z
  jboolean j_face_det_initialized =
      env->GetBooleanField(j_face_det_result_obj, j_face_det_initialized_id_cc);
  if (j_face_det_initialized == JNI_FALSE) {
    return false;
  }

  jobjectArray j_face_det_boxes_float_arr = reinterpret_cast<jobjectArray>(
      env->GetObjectField(j_face_det_result_obj, j_face_det_boxes_id_cc));
  jfloatArray j_face_det_scores_float_arr = reinterpret_cast<jfloatArray>(
      env->GetObjectField(j_face_det_result_obj, j_face_det_scores_id_cc));
  jint j_landmarks_per_face = env->GetIntField(
      j_face_det_result_obj, j_face_det_landmarks_per_face_id_cc);

  int len = env->GetArrayLength(j_face_det_boxes_float_arr);
  if ((len == 0) || (len != env->GetArrayLength(
      j_face_det_scores_float_arr))) {
    return false;
  }

  // Init Cxx result
  c_result_ptr->Clear();
  // Set landmarks_per_face before Resize
  c_result_ptr->landmarks_per_face = j_landmarks_per_face;
  c_result_ptr->Resize(len);

  // mBoxes float[][] shape (n,4): [[F
  bool c_check_validation = true;
  for (int i = 0; i < len; ++i) {
    auto j_box = reinterpret_cast<jfloatArray>(
        env->GetObjectArrayElement(j_face_det_boxes_float_arr, i));
    if (env->GetArrayLength(j_box) == 4) {
      jfloat *j_box_ptr = env->GetFloatArrayElements(j_box, nullptr);
      std::memcpy(c_result_ptr->boxes[i].data(), j_box_ptr, 4 * sizeof(float));
      env->ReleaseFloatArrayElements(j_box, j_box_ptr, 0);
    } else {
      c_check_validation = false;
      break;
    }
  }
  if (!c_check_validation) {
    LOGE("The length of each detection box is not equal 4!");
    return false;
  }

  // mScores float[]  shape (n):   [F
  jfloat *j_face_det_scores_ptr =
      env->GetFloatArrayElements(j_face_det_scores_float_arr, nullptr);
  std::memcpy(c_result_ptr->scores.data(), j_face_det_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(j_face_det_scores_float_arr, j_face_det_scores_ptr, 0);

  // mLandmarks float[][] shape (n,2): [[F
  if (j_landmarks_per_face > 0) {
    jobjectArray j_face_det_landmarks_float_arr = reinterpret_cast<jobjectArray>(
        env->GetObjectField(j_face_det_result_obj, j_face_det_landmarks_id_cc));
    const int landmarks_len = env->GetArrayLength(j_face_det_landmarks_float_arr);

    for (int i = 0; i < landmarks_len; ++i) {
      auto j_landmark = reinterpret_cast<jfloatArray>(
          env->GetObjectArrayElement(j_face_det_landmarks_float_arr, i));
      if (env->GetArrayLength(j_landmark) == 2) {
        jfloat *j_landmark_ptr = env->GetFloatArrayElements(j_landmark, nullptr);
        std::memcpy(c_result_ptr->landmarks[i].data(), j_landmark_ptr, 2 * sizeof(float));
        env->ReleaseFloatArrayElements(j_landmark, j_landmark_ptr, 0);
      } else {
        c_check_validation = false;
        break;
      }
    }
  }
  if (!c_check_validation) {
    LOGE("The length of each landmarks is not equal 2!");
    return false;
  }

  // Release local Refs
  env->DeleteLocalRef(j_face_det_result_clazz_cc);

  return true;
}

bool AllocateKeyPointDetectionResultFromJava(
    JNIEnv *env, jobject j_keypoint_det_result_obj, void *cxx_result) {
  // WARN: Please make sure 'j_keypoint_det_result_obj' param
  // is a ref of Java KeyPointDetectionResult.
  // Field signatures of Java KeyPointDetectionResult:
  // (1) mBoxes float[][] shape (n*num_joints,2): [[F
  // (2) mScores float[]  shape (n*num_joints):   [F
  // (3) mNumJoints int  shape (1):               I
  // (4) mInitialized boolean:                    Z
  // Docs: docs/api/vision_results/keypointdetection_result.md
  if (cxx_result == nullptr || j_keypoint_det_result_obj == nullptr) {
    return false;
  }
  auto c_result_ptr = reinterpret_cast<vision::KeyPointDetectionResult *>(cxx_result);

  const jclass j_keypoint_det_result_clazz_cc = env->FindClass(
      "com/baidu/paddle/fastdeploy/vision/KeyPointDetectionResult");
  const jfieldID j_keypoint_det_keypoints_id_cc = env->GetFieldID(
      j_keypoint_det_result_clazz_cc, "mKeyPoints", "[[F");
  const jfieldID j_keypoint_det_scores_id_cc = env->GetFieldID(
      j_keypoint_det_result_clazz_cc, "mScores", "[F");
  const jfieldID j_keypoint_det_num_joints_id_cc = env->GetFieldID(
      j_keypoint_det_result_clazz_cc, "mNumJoints", "I");
  const jfieldID j_keypoint_det_initialized_id_cc = env->GetFieldID(
      j_keypoint_det_result_clazz_cc, "mInitialized", "Z");

  if (!env->IsInstanceOf(j_keypoint_det_result_obj, j_keypoint_det_result_clazz_cc)) {
    return false;
  }

  // mInitialized boolean:         Z
  jboolean j_keypoint_det_initialized =
      env->GetBooleanField(j_keypoint_det_result_obj, j_keypoint_det_initialized_id_cc);
  if (j_keypoint_det_initialized == JNI_FALSE) {
    return false;
  }

  jobjectArray j_keypoint_det_keypoints_float_arr = reinterpret_cast<jobjectArray>(
      env->GetObjectField(j_keypoint_det_result_obj, j_keypoint_det_keypoints_id_cc));
  jfloatArray j_keypoint_det_scores_float_arr = reinterpret_cast<jfloatArray>(
      env->GetObjectField(j_keypoint_det_result_obj, j_keypoint_det_scores_id_cc));
  jint j_keypoint_det_num_joints = env->GetIntField(
      j_keypoint_det_result_obj, j_keypoint_det_num_joints_id_cc);

  int len = env->GetArrayLength(j_keypoint_det_keypoints_float_arr);
  if ((len == 0) || (len != env->GetArrayLength(j_keypoint_det_scores_float_arr)) ||
      (j_keypoint_det_num_joints < 0)) {
    return false;
  }

  // Init Cxx result
  c_result_ptr->Clear();

  // mKeyPoints float[][] shape (n*num_joints,2): [[F
  c_result_ptr->keypoints.resize(len);
  bool c_check_validation = true;
  for (int i = 0; i < len; ++i) {
    auto j_point = reinterpret_cast<jfloatArray>(
        env->GetObjectArrayElement(j_keypoint_det_keypoints_float_arr, i));
    if (env->GetArrayLength(j_point) == 2) {
      jfloat *j_point_ptr = env->GetFloatArrayElements(j_point, nullptr);
      std::memcpy(c_result_ptr->keypoints[i].data(), j_point_ptr, 2 * sizeof(float));
      env->ReleaseFloatArrayElements(j_point, j_point_ptr, 0);
    } else {
      c_check_validation = false;
      break;
    }
  }
  if (!c_check_validation) {
    LOGE("The length of each detection box is not equal 2!");
    return false;
  }

  // mScores float[]  shape (n):   [F
  c_result_ptr->scores.resize(len);
  jfloat *j_keypoint_det_scores_ptr =
      env->GetFloatArrayElements(j_keypoint_det_scores_float_arr, nullptr);
  std::memcpy(c_result_ptr->scores.data(), j_keypoint_det_scores_ptr, len * sizeof(float));
  env->ReleaseFloatArrayElements(j_keypoint_det_scores_float_arr, j_keypoint_det_scores_ptr, 0);

  // mNumJoints int  shape (1):   I
  c_result_ptr->num_joints = static_cast<int>(j_keypoint_det_num_joints);

  // Release local Refs
  env->DeleteLocalRef(j_keypoint_det_result_clazz_cc);

  return true;
}

bool AllocateCxxResultFromJava(
    JNIEnv *env, jobject j_result_obj, void *cxx_result,
    vision::ResultType type) {
  if (type == vision::ResultType::CLASSIFY) {
    return AllocateClassifyResultFromJava(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::DETECTION) {
    return AllocateDetectionResultFromJava(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::OCR) {
    return AllocateOCRResultFromJava(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::SEGMENTATION) {
    return AllocateSegmentationResultFromJava(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::FACE_DETECTION) {
    return AllocateFaceDetectionResultFromJava(env, j_result_obj, cxx_result);
  } else if (type == vision::ResultType::KEYPOINT_DETECTION) {
    return AllocateKeyPointDetectionResultFromJava(env, j_result_obj, cxx_result);
  } else {
    LOGE("Not support this ResultType in JNI now, type: %d",
         static_cast<int>(type));
    return false;
  }
}

}  // namespace jni
}  // namespace fastdeploy

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_SegmentationResult_releaseCxxBufferNative(
    JNIEnv *env, jobject thiz) {
  const jclass j_seg_result_clazz = env->GetObjectClass(thiz);
  const jfieldID j_enable_cxx_buffer_id = env->GetFieldID(
      j_seg_result_clazz, "mEnableCxxBuffer", "Z");
  const jfieldID  j_cxx_buffer_id = env->GetFieldID(
      j_seg_result_clazz, "mCxxBuffer", "J");
  const jfieldID j_seg_initialized_id = env->GetFieldID(
      j_seg_result_clazz, "mInitialized", "Z");

  jboolean j_enable_cxx_buffer =
      env->GetBooleanField(thiz, j_enable_cxx_buffer_id);
  if (j_enable_cxx_buffer == JNI_FALSE) {
    return JNI_FALSE;
  }
  jlong j_cxx_buffer = env->GetLongField(thiz, j_cxx_buffer_id);
  if (j_cxx_buffer == 0) {
    return JNI_FALSE;
  }
  auto c_result_ptr = reinterpret_cast<
      fastdeploy::vision::SegmentationResult *>(j_cxx_buffer);
  delete c_result_ptr;
  LOGD("[End] Release SegmentationResult & CxxBuffer in native !");

  env->SetBooleanField(thiz, j_seg_initialized_id, JNI_FALSE);
  env->DeleteLocalRef(j_seg_result_clazz);

  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif