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
namespace pipeline {

enum PPOCRVersion {
  OCR_V1 = 0,
  OCR_V2 = 1,
  OCR_V3 = 2
};

/// Handle the native PP-OCR pipeline resources.
class PPOCRHandler {
public:
  PPOCRHandler() = default;

  PPOCRHandler(fastdeploy::vision::ocr::DBDetector *det_model,
               fastdeploy::vision::ocr::Classifier *cls_model,
               fastdeploy::vision::ocr::Recognizer *rec_model,
               fastdeploy::pipeline::PPOCRv2 *ppocr_v2) :
      detector_(det_model), classifier_(cls_model),
      recognizer_(rec_model), ppocr_v2_(ppocr_v2) {
    if (detector_ != nullptr && classifier_ != nullptr
        && recognizer_ != nullptr && ppocr_v2_ != nullptr) {
      initialized_ = true;
    }
  }

  PPOCRHandler(fastdeploy::vision::ocr::DBDetector *det_model,
               fastdeploy::vision::ocr::Recognizer *rec_model,
               fastdeploy::pipeline::PPOCRv2 *ppocr_v2) :
      detector_(det_model), recognizer_(rec_model),
      ppocr_v2_(ppocr_v2) {
    if (detector_ != nullptr && recognizer_ != nullptr
        && ppocr_v2_ != nullptr) {
      initialized_ = true;
    }
  }

  PPOCRHandler(fastdeploy::vision::ocr::DBDetector *det_model,
               fastdeploy::vision::ocr::Classifier *cls_model,
               fastdeploy::vision::ocr::Recognizer *rec_model,
               fastdeploy::pipeline::PPOCRv3 *ppocr_v3) :
      detector_(det_model), classifier_(cls_model),
      recognizer_(rec_model), ppocr_v3_(ppocr_v3) {
    if (detector_ != nullptr && classifier_ != nullptr
        && recognizer_ != nullptr && ppocr_v3_ != nullptr) {
      initialized_ = true;
    }
  }

  PPOCRHandler(fastdeploy::vision::ocr::DBDetector *det_model,
               fastdeploy::vision::ocr::Recognizer *rec_model,
               fastdeploy::pipeline::PPOCRv3 *ppocr_v3) :
      detector_(det_model), recognizer_(rec_model),
      ppocr_v3_(ppocr_v3) {
    if (detector_ != nullptr && recognizer_ != nullptr
        && ppocr_v3_ != nullptr) {
      initialized_ = true;
    }
  }

  void SetPPOCRVersion(PPOCRVersion version_tag) {
    ppocr_version_tag_ = version_tag;
  }

  bool Predict(cv::Mat* img, fastdeploy::vision::OCRResult* result) {
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

  bool Initialized() {
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

  // Call init manually if you want to release the allocated
  // PP-OCRv2/v3's memory by 'new' operator via 'delete'.
  bool ReleaseAllocatedOCRMemories() {
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

  void PrintPPOCRHandlerTimeOfRuntime() const {
    if ((detector_ != nullptr) && (detector_->EnabledRecordTimeOfRuntime())) {
      auto det_info_of_runtime = detector_->PrintStatisInfoOfRuntime();
      LOGD("[Det] Avg runtime costs %f ms", det_info_of_runtime["avg_time"] * 1000.0f);
    }
    if ((classifier_ != nullptr) && (classifier_->EnabledRecordTimeOfRuntime())) {
      auto cls_info_of_runtime = classifier_->PrintStatisInfoOfRuntime();
      LOGD("[Cls] Avg runtime costs %f ms", cls_info_of_runtime["avg_time"] * 1000.0f);
    }
    if ((recognizer_ != nullptr) && (recognizer_->EnabledRecordTimeOfRuntime())) {
      auto rec_info_of_runtime = recognizer_->PrintStatisInfoOfRuntime();
      LOGD("[Rec] Avg runtime costs %f ms", rec_info_of_runtime["avg_time"] * 1000.0f);
    }
  }

public:
  fastdeploy::vision::ocr::DBDetector *detector_ = nullptr;
  fastdeploy::vision::ocr::Classifier *classifier_ = nullptr;
  fastdeploy::vision::ocr::Recognizer *recognizer_ = nullptr;
  fastdeploy::pipeline::PPOCRv2 *ppocr_v2_ = nullptr;
  fastdeploy::pipeline::PPOCRv3 *ppocr_v3_ = nullptr;

private:
  bool initialized_ = false;
  PPOCRVersion ppocr_version_tag_ = PPOCRVersion::OCR_V2;
};

}  // namespace pipeline
}  // namespace jni
}  // namespace fastdeploy

namespace fastdeploy {
namespace jni {

/// Rendering OCRResult to ARGB888Bitmap
void RenderingOCR(
    JNIEnv *env, const cv::Mat &c_bgr, const vision::OCRResult &c_result,
    jobject argb8888_bitmap, bool saved, jstring saved_image_path) {
  if (!c_result.boxes.empty()) {
    auto t = fastdeploy::jni::GetCurrentTime();
    cv::Mat c_vis_im;
    c_vis_im = fastdeploy::vision::VisOcr(c_bgr, c_result);
    LOGD("Visualize from native costs %f ms",
         fastdeploy::jni::GetElapsedTime(t));

    // Rendering to bitmap
    if (!fastdeploy::jni::BGR2ARGB888Bitmap(env, argb8888_bitmap, c_vis_im)) {
      LOGD("Write to bitmap from native failed!");
    }
    std::string c_saved_image_path =
        fastdeploy::jni::ConvertTo<std::string>(env, saved_image_path);
    if (!c_saved_image_path.empty() && saved) {
      cv::imwrite(c_saved_image_path, c_bgr);
    }
  }
}

}  // namespace jni
}  // namespace fastdeploy

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_pipeline_PPOCRBase_bindNative(
    JNIEnv *env, jobject thiz, jint ppocr_version_tag,
    jstring det_model_file, jstring det_params_file, jstring cls_model_file,
    jstring cls_params_file, jstring rec_model_file, jstring rec_params_file,
    jstring rec_label_path, jobject det_runtime_option,
    jobject cls_runtime_option, jobject rec_runtime_option,
    jboolean have_cls_model) {
  auto c_ocr_version_tag = static_cast<
      fastdeploy::jni::pipeline::PPOCRVersion>(ppocr_version_tag);
  if (c_ocr_version_tag == fastdeploy::jni::pipeline::PPOCRVersion::OCR_V1) {
    LOGE("Not support for PPOCRVersion::OCR_V1 now!");
    return 0;
  }
  // TODO(qiuyanjun): Allows users to set model parameters, such as det_db_box_thresh,
  //  det_db_thresh, use_dilation, etc. These parameters should be passed in via JNI.
  auto c_det_model_file = fastdeploy::jni::ConvertTo<std::string>(env, det_model_file);
  auto c_det_params_file = fastdeploy::jni::ConvertTo<std::string>(env, det_params_file);
  auto c_cls_model_file = fastdeploy::jni::ConvertTo<std::string>(env, cls_model_file);
  auto c_cls_params_file = fastdeploy::jni::ConvertTo<std::string>(env, cls_params_file);
  auto c_rec_model_file = fastdeploy::jni::ConvertTo<std::string>(env, rec_model_file);
  auto c_rec_params_file = fastdeploy::jni::ConvertTo<std::string>(env, rec_params_file);
  auto c_rec_label_path = fastdeploy::jni::ConvertTo<std::string>(env, rec_label_path);
  auto c_det_runtime_option = fastdeploy::jni::NewCxxRuntimeOption(env, det_runtime_option);
  auto c_cls_runtime_option = fastdeploy::jni::NewCxxRuntimeOption(env, cls_runtime_option);
  auto c_rec_runtime_option = fastdeploy::jni::NewCxxRuntimeOption(env, rec_runtime_option);
  auto c_have_cls_model = static_cast<bool>(have_cls_model);

  // Init PP-OCR pipeline
  auto c_det_model_ptr = new fastdeploy::vision::ocr::DBDetector(
      c_det_model_file, c_det_params_file, c_det_runtime_option);
  auto c_rec_model_ptr = new fastdeploy::vision::ocr::Recognizer(
      c_rec_model_file, c_rec_params_file, c_rec_label_path, c_rec_runtime_option);

  // PP-OCRv2
  if (c_ocr_version_tag == fastdeploy::jni::pipeline::PPOCRVersion::OCR_V2) {
    if (c_have_cls_model) {
      auto c_cls_model_ptr = new fastdeploy::vision::ocr::Classifier(
          c_cls_model_file, c_cls_params_file, c_cls_runtime_option);
      auto c_ppocr_pipeline_ptr = new fastdeploy::pipeline::PPOCRv2(
          c_det_model_ptr, c_cls_model_ptr, c_rec_model_ptr);
      // PP-OCRv2 handler with cls model
      auto c_ppocr_handler_ptr = new fastdeploy::jni::pipeline::PPOCRHandler(
          c_det_model_ptr, c_cls_model_ptr, c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(c_ppocr_handler_ptr);  // native handler context
    } else {
      auto c_ppocr_pipeline_ptr = new fastdeploy::pipeline::PPOCRv2(
          c_det_model_ptr, c_rec_model_ptr);
      // PP-OCRv2 handler without cls model
      auto c_ppocr_handler_ptr = new fastdeploy::jni::pipeline::PPOCRHandler(
          c_det_model_ptr, c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(c_ppocr_handler_ptr);  // native handler context
    }
  } // PP-OCRv3
  else if (c_ocr_version_tag == fastdeploy::jni::pipeline::PPOCRVersion::OCR_V3) {
    if (c_have_cls_model) {
      auto c_cls_model_ptr = new fastdeploy::vision::ocr::Classifier(
          c_cls_model_file, c_cls_params_file, c_cls_runtime_option);
      auto c_ppocr_pipeline_ptr = new fastdeploy::pipeline::PPOCRv3(
          c_det_model_ptr, c_cls_model_ptr, c_rec_model_ptr);
      // PP-OCRv3 handler with cls model
      auto c_ppocr_handler_ptr = new fastdeploy::jni::pipeline::PPOCRHandler(
          c_det_model_ptr, c_cls_model_ptr, c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(c_ppocr_handler_ptr);  // native handler context
    } else {
      auto c_ppocr_pipeline_ptr = new fastdeploy::pipeline::PPOCRv3(
          c_det_model_ptr, c_rec_model_ptr);
      // PP-OCRv3 handler without cls model
      auto c_ppocr_handler_ptr = new fastdeploy::jni::pipeline::PPOCRHandler(
          c_det_model_ptr, c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(c_ppocr_handler_ptr);  // native handler context
    }
  }
  return 0;

}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_pipeline_PPOCRBase_predictNative(
    JNIEnv *env, jobject thiz, jlong native_handler_context,
    jobject argb8888_bitmap, jboolean saved, jstring saved_image_path,
    jboolean rendering) {
  if (native_handler_context == 0) {
    return NULL;
  }
  cv::Mat c_bgr;
  if (!fastdeploy::jni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return NULL;
  }
  auto c_ppocr_handler_ptr = reinterpret_cast<
      fastdeploy::jni::pipeline::PPOCRHandler*>(
          native_handler_context);

  fastdeploy::vision::OCRResult c_result;
  c_ppocr_handler_ptr->Predict(&c_bgr, &c_result);
  // TODO(qiuyanjun): remove this info
  LOGD("Result: %s", c_result.Str().c_str());
  c_ppocr_handler_ptr->PrintPPOCRHandlerTimeOfRuntime();

  if (rendering) {
    fastdeploy::jni::RenderingOCR(env, c_bgr, c_result,
                                  argb8888_bitmap, saved,
                                  saved_image_path);
  }

  return fastdeploy::jni::NewJavaResultFromCxx(
      env, reinterpret_cast<void *>(&c_result),
      fastdeploy::vision::ResultType::OCR);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_pipeline_PPOCRBase_releaseNative(
    JNIEnv *env, jobject thiz, jlong native_handler_context) {
  if (native_handler_context == 0) {
    return JNI_FALSE;
  }
  auto c_ppocr_handler_ptr = reinterpret_cast<
      fastdeploy::jni::pipeline::PPOCRHandler*>(
          native_handler_context);
  if (!c_ppocr_handler_ptr->ReleaseAllocatedOCRMemories()) {
    delete c_ppocr_handler_ptr;
    return JNI_FALSE;
  }
  delete c_ppocr_handler_ptr;
  LOGD("[End] Release PPOCRHandler in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif
