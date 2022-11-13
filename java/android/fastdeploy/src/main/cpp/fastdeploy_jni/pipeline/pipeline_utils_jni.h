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

#pragma once

#include <jni.h>  // NOLINT
#include "fastdeploy/vision.h"  // NOLINT
#include "fastdeploy_jni/perf_jni.h"  // NOLINT
#include "fastdeploy_jni/bitmap_jni.h"  // NOLINT

namespace fastdeploy {
namespace jni {

enum PPOCRVersion { OCR_V1 = 0, OCR_V2 = 1, OCR_V3 = 2 };

/// Handle the native PP-OCR pipeline resources.
class PPOCRHandler {
 public:
  PPOCRHandler() = default;

  PPOCRHandler(vision::ocr::DBDetector *det_model,
               vision::ocr::Classifier *cls_model,
               vision::ocr::Recognizer *rec_model,
               pipeline::PPOCRv2 *ppocr_v2);

  PPOCRHandler(vision::ocr::DBDetector *det_model,
               vision::ocr::Recognizer *rec_model,
               pipeline::PPOCRv2 *ppocr_v2);

  PPOCRHandler(vision::ocr::DBDetector *det_model,
               vision::ocr::Classifier *cls_model,
               vision::ocr::Recognizer *rec_model,
               pipeline::PPOCRv3 *ppocr_v3);

  PPOCRHandler(vision::ocr::DBDetector *det_model,
               vision::ocr::Recognizer *rec_model,
               pipeline::PPOCRv3 *ppocr_v3);

  void SetPPOCRVersion(PPOCRVersion version_tag);

  bool Predict(cv::Mat *img, vision::OCRResult *result);

  bool Initialized();

  // Call init manually if you want to release the allocated
  // PP-OCRv2/v3's memory by 'new' operator via 'delete'.
  bool ReleaseAllocatedOCRMemories();

 public:
  vision::ocr::DBDetector *detector_ = nullptr;
  vision::ocr::Classifier *classifier_ = nullptr;
  vision::ocr::Recognizer *recognizer_ = nullptr;
  pipeline::PPOCRv2 *ppocr_v2_ = nullptr;
  pipeline::PPOCRv3 *ppocr_v3_ = nullptr;

 private:
  bool initialized_ = false;
  PPOCRVersion ppocr_version_tag_ = PPOCRVersion::OCR_V2;
};

void RenderingOCR(JNIEnv *env, const cv::Mat &c_bgr,
                  const vision::OCRResult &c_result,
                  jobject argb8888_bitmap, bool save_image,
                  jstring saved_path);

}  // namespace jni
}  // namespace fastdeploy
