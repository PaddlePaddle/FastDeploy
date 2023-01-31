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

#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/assets_loader_jni.h" // NOLINT
#include "fastdeploy_jni/vision/classification/classification_utils_jni.h"

namespace fastdeploy {
namespace jni {

/// Rendering ClassifyResult to ARGB888Bitmap
void RenderingClassify(JNIEnv *env, const cv::Mat &c_bgr,
                       const vision::ClassifyResult &c_result,
                       jobject argb8888_bitmap, bool save_image,
                       float score_threshold, jstring save_path) {
  if (!c_result.scores.empty()) {
    auto t = GetCurrentTime();
    cv::Mat c_vis_im;
    if (AssetsLoader::IsClassificationLabelsLoaded()) {
      c_vis_im = vision::VisClassification(
          c_bgr, c_result, AssetsLoader::GetClassificationLabels(), 5,
          score_threshold, 1.0f);
    } else {
      c_vis_im =
          vision::VisClassification(c_bgr, c_result, 5, score_threshold, 1.0f);
    }
    LOGD("Visualize from native costs %f ms", GetElapsedTime(t));

    if (!BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      LOGD("Write to bitmap from native failed!");
    }
    auto c_saved_image_path = ConvertTo<std::string>(env, save_path);
    if (!c_saved_image_path.empty() && save_image) {
      cv::imwrite(c_saved_image_path, c_bgr);
    }
  }
}

}  // namespace jni
}  // namespace fastdeploy
