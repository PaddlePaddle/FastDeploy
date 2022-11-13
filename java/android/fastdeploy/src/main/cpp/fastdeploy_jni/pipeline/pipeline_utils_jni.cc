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
#include "fastdeploy_jni/pipeline/pipeline_utils_jni.h"  // NOLINT

namespace fastdeploy {
namespace jni {

/// Handle the native PP-OCR pipeline resources.
PPOCRHandler::PPOCRHandler(vision::ocr::DBDetector *det_model,
                           vision::ocr::Classifier *cls_model,
                           vision::ocr::Recognizer *rec_model,
                           pipeline::PPOCRv2 *ppocr_v2)
    : detector_(det_model),
      classifier_(cls_model),
      recognizer_(rec_model),
      ppocr_v2_(ppocr_v2) {
  if (detector_ != nullptr && classifier_ != nullptr &&
      recognizer_ != nullptr && ppocr_v2_ != nullptr) {
    initialized_ = true;
  }
}

PPOCRHandler::PPOCRHandler(vision::ocr::DBDetector *det_model,
                           vision::ocr::Recognizer *rec_model,
                           pipeline::PPOCRv2 *ppocr_v2)
    : detector_(det_model), recognizer_(rec_model), ppocr_v2_(ppocr_v2) {
  if (detector_ != nullptr && recognizer_ != nullptr && ppocr_v2_ != nullptr) {
    initialized_ = true;
  }
}

PPOCRHandler::PPOCRHandler(vision::ocr::DBDetector *det_model,
                           vision::ocr::Classifier *cls_model,
                           vision::ocr::Recognizer *rec_model,
                           pipeline::PPOCRv3 *ppocr_v3)
    : detector_(det_model),
      classifier_(cls_model),
      recognizer_(rec_model),
      ppocr_v3_(ppocr_v3) {
  if (detector_ != nullptr && classifier_ != nullptr &&
      recognizer_ != nullptr && ppocr_v3_ != nullptr) {
    initialized_ = true;
  }
}

PPOCRHandler::PPOCRHandler(vision::ocr::DBDetector *det_model,
                           vision::ocr::Recognizer *rec_model,
                           pipeline::PPOCRv3 *ppocr_v3)
    : detector_(det_model), recognizer_(rec_model), ppocr_v3_(ppocr_v3) {
  if (detector_ != nullptr && recognizer_ != nullptr && ppocr_v3_ != nullptr) {
    initialized_ = true;
  }
}

void PPOCRHandler::SetPPOCRVersion(PPOCRVersion version_tag) {
  ppocr_version_tag_ = version_tag;
}

bool PPOCRHandler::Predict(cv::Mat *img, vision::OCRResult *result) {
  if (ppocr_version_tag_ == PPOCRVersion::OCR_V2) {
    if (ppocr_v2_ != nullptr) {
      return ppocr_v2_->Predict(img, result);
    }
    return false;
  } else if (ppocr_version_tag_ == PPOCRVersion::OCR_V3) {
    if (ppocr_v3_ != nullptr) {
      return ppocr_v3_->Predict(img, result);
    }
    return false;
  }
  return false;
}

bool PPOCRHandler::Initialized() {
  if (!initialized_) {
    return false;
  }
  if (ppocr_version_tag_ == PPOCRVersion::OCR_V2) {
    if (ppocr_v2_ != nullptr) {
      return ppocr_v2_->Initialized();
    }
    return false;
  } else if (ppocr_version_tag_ == PPOCRVersion::OCR_V3) {
    if (ppocr_v3_ != nullptr) {
      return ppocr_v3_->Initialized();
    }
    return false;
  }
  return false;
}

bool PPOCRHandler::ReleaseAllocatedOCRMemories() {
  if (!Initialized()) {
    return false;
  }
  if (detector_ != nullptr) {
    delete detector_;
    detector_ = nullptr;
    LOGD("[End] Release DBDetector in native !");
  }
  if (classifier_ != nullptr) {
    delete classifier_;
    classifier_ = nullptr;
    LOGD("[End] Release Classifier in native !");
  }
  if (recognizer_ != nullptr) {
    delete recognizer_;
    recognizer_ = nullptr;
    LOGD("[End] Release Recognizer in native !");
  }
  if (ppocr_v2_ != nullptr) {
    delete ppocr_v2_;
    ppocr_v2_ = nullptr;
    LOGD("[End] Release PP-OCRv2 in native !");
  }
  if (ppocr_v3_ != nullptr) {
    delete ppocr_v3_;
    ppocr_v3_ = nullptr;
    LOGD("[End] Release PP-OCRv3 in native !");
  }
  initialized_ = false;
  return true;
}

/// Rendering OCRResult to ARGB888Bitmap
void RenderingOCR(JNIEnv *env, const cv::Mat &c_bgr,
                  const vision::OCRResult &c_result, jobject argb8888_bitmap,
                  bool save_image, jstring saved_path) {
  if (!c_result.boxes.empty()) {
    auto t = GetCurrentTime();
    cv::Mat c_vis_im;
    c_vis_im = vision::VisOcr(c_bgr, c_result);
    LOGD("Visualize from native costs %f ms", GetElapsedTime(t));

    if (!BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      LOGD("Write to bitmap from native failed!");
    }
    auto c_saved_image_path = ConvertTo<std::string>(env, saved_path);
    if (!c_saved_image_path.empty() && save_image) {
      cv::imwrite(c_saved_image_path, c_bgr);
    }
  }
}

}  // namespace jni
}  // namespace fastdeploy
