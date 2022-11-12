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

/// Rendering SegmentationResult to ARGB888Bitmap
void RenderingSegmentation(
    JNIEnv *env, const cv::Mat &c_bgr, const vision::SegmentationResult &c_result,
    jobject argb8888_bitmap, bool save_image, float weight,
    jstring save_path) {
  if (!c_result.label_map.empty()) {
    auto t = fastdeploy::jni::GetCurrentTime();

    auto c_vis_im = fastdeploy::vision::VisSegmentation(c_bgr, c_result, weight);
    LOGD("Visualize from native costs %f ms",
         fastdeploy::jni::GetElapsedTime(t));

    if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      LOGD("Write to bitmap from native failed!");
    }
    std::string c_saved_image_path =
        fastdeploy::jni::ConvertTo<std::string>(env, save_path);
    if (!c_saved_image_path.empty() && save_image) {
      cv::imwrite(c_saved_image_path, c_vis_im);
    }
  }
}

/// Show the time cost of Runtime
void PrintPPSegTimeOfRuntime(
    vision::segmentation::PaddleSegModel *c_model_ptr,
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
Java_com_baidu_paddle_fastdeploy_vision_segmentation_PaddleSegModel_bindNative(
    JNIEnv *env, jobject thiz, jstring model_file, jstring params_file,
    jstring config_file, jobject runtime_option) {
  // TODO: implement bindNative()
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_vision_segmentation_PaddleSegModel_predictNative(
    JNIEnv *env, jobject thiz, jlong cxx_context, jobject argb8888_bitmap,
    jboolean save_image, jstring save_path, jboolean rendering, jfloat weight) {
  // TODO: implement predictNative()
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_segmentation_PaddleSegModel_releaseNative(
    JNIEnv *env, jobject thiz, jlong cxx_context) {
  // TODO: implement releaseNative()
}

#ifdef __cplusplus
}
#endif
