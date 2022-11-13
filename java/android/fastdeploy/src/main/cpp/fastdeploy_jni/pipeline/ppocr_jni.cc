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
#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/assets_loader_jni.h" // NOLINT
#include "fastdeploy_jni/runtime_option_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT
#include "fastdeploy_jni/pipeline/pipeline_utils_jni.h"  // NOLINT

namespace fni = fastdeploy::jni;
namespace vision = fastdeploy::vision;
namespace ocr = fastdeploy::vision::ocr;
namespace pipeline = fastdeploy::pipeline;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_pipeline_PPOCRBase_bindNative(
    JNIEnv *env, jobject thiz, jint ppocr_version_tag, jstring det_model_file,
    jstring det_params_file, jstring cls_model_file, jstring cls_params_file,
    jstring rec_model_file, jstring rec_params_file, jstring rec_label_path,
    jobject det_runtime_option, jobject cls_runtime_option,
    jobject rec_runtime_option, jboolean have_cls_model) {
  auto c_ocr_version_tag = static_cast<fni::PPOCRVersion>(ppocr_version_tag);
  if (c_ocr_version_tag == fni::PPOCRVersion::OCR_V1) {
    LOGE("Not support for PPOCRVersion::OCR_V1 now!");
    return 0;
  }
  // TODO(qiuyanjun): Allows users to set model parameters, such as
  // det_db_box_thresh, det_db_thresh, use_dilation, etc. These
  // parameters should be passed in via JNI.
  auto c_det_model_file = fni::ConvertTo<std::string>(env, det_model_file);
  auto c_det_params_file = fni::ConvertTo<std::string>(env, det_params_file);
  auto c_cls_model_file = fni::ConvertTo<std::string>(env, cls_model_file);
  auto c_cls_params_file = fni::ConvertTo<std::string>(env, cls_params_file);
  auto c_rec_model_file = fni::ConvertTo<std::string>(env, rec_model_file);
  auto c_rec_params_file = fni::ConvertTo<std::string>(env, rec_params_file);
  auto c_rec_label_path = fni::ConvertTo<std::string>(env, rec_label_path);
  auto c_det_runtime_option = fni::NewCxxRuntimeOption(env, det_runtime_option);
  auto c_cls_runtime_option = fni::NewCxxRuntimeOption(env, cls_runtime_option);
  auto c_rec_runtime_option = fni::NewCxxRuntimeOption(env, rec_runtime_option);
  auto c_have_cls_model = static_cast<bool>(have_cls_model);

  // Init PP-OCR pipeline
  auto c_det_model_ptr = new ocr::DBDetector(
      c_det_model_file, c_det_params_file, c_det_runtime_option);
  INITIALIZED_OR_RETURN(c_det_model_ptr)

  auto c_rec_model_ptr = new ocr::Recognizer(
      c_rec_model_file, c_rec_params_file, c_rec_label_path, c_rec_runtime_option);
  INITIALIZED_OR_RETURN(c_rec_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
  c_det_model_ptr->EnableRecordTimeOfRuntime();
  c_rec_model_ptr->EnableRecordTimeOfRuntime();
#endif
  // PP-OCRv2
  if (c_ocr_version_tag == fni::PPOCRVersion::OCR_V2) {
    if (c_have_cls_model) {
      auto c_cls_model_ptr = new ocr::Classifier(
          c_cls_model_file, c_cls_params_file, c_cls_runtime_option);
      INITIALIZED_OR_RETURN(c_cls_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
      c_cls_model_ptr->EnableRecordTimeOfRuntime();
#endif
      auto c_ppocr_pipeline_ptr = new pipeline::PPOCRv2(
          c_det_model_ptr, c_cls_model_ptr, c_rec_model_ptr);
      // PP-OCRv2 handler with cls model
      auto c_ppocr_handler_ptr =
          new fni::PPOCRHandler(c_det_model_ptr, c_cls_model_ptr,
                                c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(
          c_ppocr_handler_ptr);  // native handler context
    } else {
      auto c_ppocr_pipeline_ptr =
          new pipeline::PPOCRv2(c_det_model_ptr, c_rec_model_ptr);
      // PP-OCRv2 handler without cls model
      auto c_ppocr_handler_ptr = new fni::PPOCRHandler(
          c_det_model_ptr, c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(
          c_ppocr_handler_ptr);  // native handler context
    }
  }  // PP-OCRv3
  else if (c_ocr_version_tag == fni::PPOCRVersion::OCR_V3) {
    if (c_have_cls_model) {
      auto c_cls_model_ptr = new ocr::Classifier(
          c_cls_model_file, c_cls_params_file, c_cls_runtime_option);
      INITIALIZED_OR_RETURN(c_cls_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
      c_cls_model_ptr->EnableRecordTimeOfRuntime();
#endif
      auto c_ppocr_pipeline_ptr = new pipeline::PPOCRv3(
          c_det_model_ptr, c_cls_model_ptr, c_rec_model_ptr);
      // PP-OCRv3 handler with cls model
      auto c_ppocr_handler_ptr =
          new fni::PPOCRHandler(c_det_model_ptr, c_cls_model_ptr,
                                c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(
          c_ppocr_handler_ptr);  // native handler context
    } else {
      auto c_ppocr_pipeline_ptr =
          new pipeline::PPOCRv3(c_det_model_ptr, c_rec_model_ptr);
      // PP-OCRv3 handler without cls model
      auto c_ppocr_handler_ptr = new fni::PPOCRHandler(
          c_det_model_ptr, c_rec_model_ptr, c_ppocr_pipeline_ptr);
      c_ppocr_handler_ptr->SetPPOCRVersion(c_ocr_version_tag);
      // WARN: need to release manually in Java !
      return reinterpret_cast<jlong>(
          c_ppocr_handler_ptr);  // native handler context
    }
  }
  return 0;
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_pipeline_PPOCRBase_predictNative(
    JNIEnv *env, jobject thiz, jlong cxx_context,
    jobject argb8888_bitmap, jboolean save_image,
    jstring save_path, jboolean rendering) {
  if (cxx_context == 0) {
    return NULL;
  }
  cv::Mat c_bgr;
  if (!fni::ARGB888Bitmap2BGR(env, argb8888_bitmap, &c_bgr)) {
    return NULL;
  }
  auto c_ppocr_handler_ptr = reinterpret_cast<fni::PPOCRHandler *>(cxx_context);

  vision::OCRResult c_result;
  c_ppocr_handler_ptr->Predict(&c_bgr, &c_result);
  LOGD("OCR Result: %s", c_result.Str().c_str());
  PERF_TIME_OF_RUNTIME(c_ppocr_handler_ptr->detector_, -1)
  PERF_TIME_OF_RUNTIME(c_ppocr_handler_ptr->classifier_, -1)
  PERF_TIME_OF_RUNTIME(c_ppocr_handler_ptr->recognizer_, -1)

  if (rendering) {
    fni::RenderingOCR(env, c_bgr, c_result, argb8888_bitmap,
                      save_image, save_path);
  }

  return fni::NewJavaResultFromCxx(env, reinterpret_cast<void *>(&c_result),
                                   vision::ResultType::OCR);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_pipeline_PPOCRBase_releaseNative(
    JNIEnv *env, jobject thiz, jlong cxx_context) {
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_ppocr_handler_ptr = reinterpret_cast<fni::PPOCRHandler *>(cxx_context);
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
