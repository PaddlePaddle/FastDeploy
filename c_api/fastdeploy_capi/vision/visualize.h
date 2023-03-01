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

#pragma once

#include "fastdeploy_capi/core/fd_common.h"
#include "fastdeploy_capi/core/fd_type.h"
#include "fastdeploy_capi/vision/result.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Show the visualized results for detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the result produced by model
 * \param[in] score_threshold threshold for result scores, the bounding box will not be shown if the score is less than score_threshold
 * \param[in] line_size line size for bounding boxes
 * \param[in] font_size font size for text
 * \return cv::Mat type stores the visualized results
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat
FD_C_VisDetection(FD_C_Mat im, FD_C_DetectionResult* fd_detection_result,
                  float score_threshold, int line_size, float font_size);

/** \brief Show the visualized results with custom labels for detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the result produced by model
 * \param[in] labels the visualized result will show the bounding box contain class label
 * \param[in] score_threshold threshold for result scores, the bounding box will not be shown if the score is less than score_threshold
 * \param[in] line_size line size for bounding boxes
 * \param[in] font_size font size for text
 * \return cv::Mat type stores the visualized results
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat FD_C_VisDetectionWithLabel(
                                     FD_C_Mat im,
                                     FD_C_DetectionResult* fd_detection_result,
                                     FD_C_OneDimArrayCstr* labels,
                                     float score_threshold,
                                     int line_size, float font_size);

/** \brief Show the visualized results for classification models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the result produced by model
 * \param[in] top_k the length of return values, e.g., if topk==2, the result will include the 2 most possible class label for input image.
 * \param[in] score_threshold threshold for top_k scores, the class will not be shown if the score is less than score_threshold
 * \param[in] font_size font size
 * \return cv::Mat type stores the visualized results
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat FD_C_VisClassification(
  FD_C_Mat im, FD_C_ClassifyResult* result, int top_k,
  float score_threshold, float font_size);
/** \brief Show the visualized results with custom labels for classification models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the result produced by model
 * \param[in] labels custom labels for user, the visualized result will show the corresponding custom labels
 * \param[in] top_k the length of return values, e.g., if topk==2, the result will include the 2 most possible class label for input image.
 * \param[in] score_threshold threshold for top_k scores, the class will not be shown if the score is less than score_threshold
 * \param[in] font_size font size
 * \return cv::Mat type stores the visualized results
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat FD_C_VisClassificationWithLabel(
  FD_C_Mat im, FD_C_ClassifyResult* result,
  FD_C_OneDimArrayCstr* labels, int top_k,
  float score_threshold, float font_size);


/** \brief Show the visualized results for Ocr models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the result produced by model
 * \return cv::Mat type stores the visualized results
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat FD_C_VisOcr(FD_C_Mat im, FD_C_OCRResult* ocr_result);

/** \brief Show the visualized results for segmentation models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the result produced by model
 * \param[in] weight transparent weight of visualized result image
 * \return cv::Mat type stores the visualized results
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat FD_C_VisSegmentation(FD_C_Mat im,
                                        FD_C_SegmentationResult* result,
                                        float weight);


#ifdef __cplusplus
}  // extern "C"
#endif
