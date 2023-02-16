// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy_capi/vision/detection/ppdet/model.h"

#include "fastdeploy_capi/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

#ifdef __cplusplus
extern "C" {
#endif

// Recognizer

FD_C_RecognizerWrapper* FD_C_CreatesRecognizerWrapper(
    const char* model_file, const char* params_file, const char* label_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {}

OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(Recognizer,
                                                   fd_c_recognizer_wrapper)

FD_C_RecognizerWrapperPredict(FD_C_RecognizerWrapper* fd_c_recognizer_wrapper,
                              FD_C_Mat img, FD_C_Cstr* text, float* rec_score) {

}

OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(Recognizer,
                                               fd_c_recognizer_wrapper)

FD_C_Bool FD_C_RecognizerWrapperBatchPredict(
    FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayCstr* texts, FD_C_OneDimArrayFloat* rec_scores) {}

FD_C_Bool FD_C_RecognizerWrapperBatchPredictWithIndex(
    FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayCstr* texts, FD_C_OneDimArrayFloat* rec_scores,
    size_t start_index, size_t end_index, FD_C_OneDimArrayInt32 indices) {}

// Classifier

FD_C_ClassifierWrapper* FD_C_CreatesClassifierWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {}

OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(Classifier,
                                                   fd_c_classifier_wrapper)

FD_C_ClassifierWrapperPredict(FD_C_ClassifierWrapper* fd_c_classifier_wrapper,
                              FD_C_Mat img, int32_t* cls_label,
                              float* cls_score) {}

OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(Classifier,
                                               fd_c_classifier_wrapper)

FD_C_ClassifierWrapperBatchPredict(
    FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayInt32* cls_labels, FD_C_OneDimArrayFloat* cls_scores) {}

FD_C_Bool FD_C_ClassifierWrapperBatchPredictWithIndex(
    FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayInt32* cls_labels, FD_C_OneDimArrayFloat* cls_scores,
    size_t start_index, size_t end_index) {}

// DBDetector
FD_C_DBDetectorWrapper* FD_C_CreatesDBDetectorWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {}

OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(DBDetector,
                                                   fd_c_dbdetector_wrapper)

FD_C_Bool FD_C_DBDetectorWrapperPredict(
    FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper, FD_C_Mat img,
    FD_C_TwoDimArrayInt32* boxes_result) {}

OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(DBDetector,
                                               fd_c_dbdetector_wrapper)

FD_C_Bool FD_C_DBDetectorWrapperBatchPredict(
    FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper, FD_C_OneDimMat imgs,
    FD_C_ThreeDimArrayInt32* det_results) {}

// PPOCRv2

FD_C_PPOCRv2Wrapper* FD_C_CreatesPPOCRv2Wrapper(
    FD_C_DBDetectorWrapper* det_model, FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model) {}

PIPELINE_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPOCRv2,
                                                        fd_c_ppocrv2_wrapper)

FD_C_Bool FD_C_PPOCRv2WrapperPredict(FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper,
                                     FD_C_Mat img, FD_C_OCRResult* result) {}

PIPELINE_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPOCRv2,
                                                    fd_c_ppocrv2_wrapper)

FD_C_DetectionResult* FD_C_OCRResultToC(
    fastdeploy::vision::OCRResult* ocr_result) {
  // Internal use, transfer fastdeploy::vision::OCRResult to
  // FD_C_OCRResult
  // FD_C_DetectionResult* fd_c_detection_result = new FD_C_DetectionResult();
  // // copy boxes
  // const int boxes_coordinate_dim = 4;
  // fd_c_detection_result->boxes.size = detection_result->boxes.size();
  // fd_c_detection_result->boxes.data =
  //     new FD_C_OneDimArrayFloat[fd_c_detection_result->boxes.size];
  // for (size_t i = 0; i < detection_result->boxes.size(); i++) {
  //   fd_c_detection_result->boxes.data[i].size = boxes_coordinate_dim;
  //   fd_c_detection_result->boxes.data[i].data = new
  //   float[boxes_coordinate_dim]; for (size_t j = 0; j < boxes_coordinate_dim;
  //   j++) {
  //     fd_c_detection_result->boxes.data[i].data[j] =
  //         detection_result->boxes[i][j];
  //   }
  // }
  // // copy scores
  // fd_c_detection_result->scores.size = detection_result->scores.size();
  // fd_c_detection_result->scores.data =
  //     new float[fd_c_detection_result->scores.size];
  // memcpy(fd_c_detection_result->scores.data, detection_result->scores.data(),
  //        sizeof(float) * fd_c_detection_result->scores.size);
  // // copy label_ids
  // fd_c_detection_result->label_ids.size = detection_result->label_ids.size();
  // fd_c_detection_result->label_ids.data =
  //     new int32_t[fd_c_detection_result->label_ids.size];
  // memcpy(fd_c_detection_result->label_ids.data,
  //        detection_result->label_ids.data(),
  //        sizeof(int32_t) * fd_c_detection_result->label_ids.size);
  // // copy masks
  // fd_c_detection_result->masks.size = detection_result->masks.size();
  // fd_c_detection_result->masks.data =
  //     new FD_C_Mask[fd_c_detection_result->masks.size];
  // for (size_t i = 0; i < detection_result->masks.size(); i++) {
  //   // copy data in mask
  //   fd_c_detection_result->masks.data[i].data.size =
  //       detection_result->masks[i].data.size();
  //   fd_c_detection_result->masks.data[i].data.data =
  //       new uint8_t[detection_result->masks[i].data.size()];
  //   memcpy(fd_c_detection_result->masks.data[i].data.data,
  //          detection_result->masks[i].data.data(),
  //          sizeof(uint8_t) * detection_result->masks[i].data.size());
  //   // copy shape in mask
  //   fd_c_detection_result->masks.data[i].shape.size =
  //       detection_result->masks[i].shape.size();
  //   fd_c_detection_result->masks.data[i].shape.data =
  //       new int64_t[detection_result->masks[i].shape.size()];
  //   memcpy(fd_c_detection_result->masks.data[i].shape.data,
  //          detection_result->masks[i].shape.data(),
  //          sizeof(int64_t) * detection_result->masks[i].shape.size());
  //   fd_c_detection_result->masks.data[i].type =
  //       static_cast<FD_C_ResultType>(detection_result->masks[i].type);
  // }
  // fd_c_detection_result->contain_masks = detection_result->contain_masks;
  // fd_c_detection_result->type =
  //     static_cast<FD_C_ResultType>(detection_result->type);
  // return fd_c_detection_result;
}

FD_C_Bool FD_C_PPOCRv2WrapperBatchPredict(
    FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* batch_result) {}

// PPOCRv3

FD_C_PPOCRv3Wrapper* FD_C_CreatesPPOCRv3Wrapper(
    FD_C_DBDetectorWrapper* det_model, FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model) {}

PIPELINE_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPOCRv3,
                                                        fd_c_ppocrv3_wrapper)

FD_C_Bool FD_C_PPOCRv3WrapperPredict(FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper,
                                     FD_C_Mat img, FD_C_OCRResult* result) {}

PIPELINE_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPOCRv3,
                                                    fd_c_ppocrv3_wrapper)

FD_C_Bool FD_C_PPOCRv3WrapperBatchPredict(
    FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* batch_result) {}

#ifdef __cplusplus
}
#endif