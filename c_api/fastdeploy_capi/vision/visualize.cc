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

#include "fastdeploy_capi/vision/visualize.h"

#include "fastdeploy/vision/visualize/visualize.h"
#include "fastdeploy_capi/internal/types_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

FD_C_Mat FD_C_VisDetection(FD_C_Mat im,
                           FD_C_DetectionResult* fd_c_detection_result,
                           float score_threshold, int line_size,
                           float font_size) {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      FD_C_CreateDetectionResultWrapperFromCResult(fd_c_detection_result);
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  cv::Mat result = fastdeploy::vision::VisDetection(
      *(reinterpret_cast<cv::Mat*>(im)), *detection_result, score_threshold,
      line_size, font_size);
  FD_C_DestroyDetectionResultWrapper(fd_c_detection_result_wrapper);
  return new cv::Mat(result);
}

FD_C_Mat FD_C_VisDetectionWithLabel(FD_C_Mat im,
                                    FD_C_DetectionResult* fd_c_detection_result,
                                    FD_C_OneDimArrayCstr* labels,
                                    float score_threshold, int line_size,
                                    float font_size) {
  std::vector<std::string> labels_in;
  for (int i = 0; i < labels->size; i++) {
    labels_in.emplace_back(labels->data[i].data);
  }
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      FD_C_CreateDetectionResultWrapperFromCResult(fd_c_detection_result);
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  cv::Mat result = fastdeploy::vision::VisDetection(
      *(reinterpret_cast<cv::Mat*>(im)), *detection_result, labels_in,
      score_threshold, line_size, font_size);
  FD_C_DestroyDetectionResultWrapper(fd_c_detection_result_wrapper);
  return new cv::Mat(result);
}

FD_C_Mat FD_C_VisClassification(FD_C_Mat im,
                                FD_C_ClassifyResult* fd_c_classify_result,
                                int top_k, float score_threshold,
                                float font_size) {
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      FD_C_CreateClassifyResultWrapperFromCResult(fd_c_classify_result);
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  cv::Mat result = fastdeploy::vision::VisClassification(
      *(reinterpret_cast<cv::Mat*>(im)), *classify_result, top_k,
      score_threshold, font_size);
  FD_C_DestroyClassifyResultWrapper(fd_c_classify_result_wrapper);
  return new cv::Mat(result);
}

FD_C_Mat FD_C_VisClassificationWithLabel(
    FD_C_Mat im, FD_C_ClassifyResult* fd_c_classify_result,
    FD_C_OneDimArrayCstr* labels, int top_k, float score_threshold,
    float font_size) {
  std::vector<std::string> labels_in;
  for (int i = 0; i < labels->size; i++) {
    labels_in.emplace_back(labels->data[i].data);
  }
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      FD_C_CreateClassifyResultWrapperFromCResult(fd_c_classify_result);
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  cv::Mat result = fastdeploy::vision::VisClassification(
      *(reinterpret_cast<cv::Mat*>(im)), *classify_result, labels_in, top_k,
      score_threshold, font_size);
  FD_C_DestroyClassifyResultWrapper(fd_c_classify_result_wrapper);
  return new cv::Mat(result);
}

FD_C_Mat FD_C_VisOcr(FD_C_Mat im, FD_C_OCRResult* fd_c_ocr_result) {
  FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper =
      FD_C_CreateOCRResultWrapperFromCResult(fd_c_ocr_result);
  auto& ocr_result =
      CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, fd_c_ocr_result_wrapper);
  cv::Mat result = fastdeploy::vision::VisOcr(*(reinterpret_cast<cv::Mat*>(im)),
                                              *ocr_result);
  FD_C_DestroyOCRResultWrapper(fd_c_ocr_result_wrapper);
  return new cv::Mat(result);
}

FD_C_Mat FD_C_VisSegmentation(FD_C_Mat im,
                              FD_C_SegmentationResult* fd_c_segmenation_result,
                              float weight) {
  FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper =
      FD_C_CreateSegmentationResultWrapperFromCResult(fd_c_segmenation_result);
  auto& segmentation_result = CHECK_AND_CONVERT_FD_TYPE(
      SegmentationResultWrapper, fd_c_segmentation_result_wrapper);
  cv::Mat result = fastdeploy::vision::VisSegmentation(
      *(reinterpret_cast<cv::Mat*>(im)), *segmentation_result, weight);
  FD_C_DestroySegmentationResultWrapper(fd_c_segmentation_result_wrapper);
  return new cv::Mat(result);
}

#ifdef __cplusplus
}
#endif
