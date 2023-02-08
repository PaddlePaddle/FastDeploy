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
#include "fastdeploy_capi/types_internal.h"

extern "C" {

FD_C_Mat FD_C_VisDetection(FD_C_Mat im,
                           FD_C_DetectionResult* fd_c_detection_result,
                           float score_threshold, int line_size,
                           float font_size) {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      FD_C_CreateDetectionResultWrapperFromData(fd_c_detection_result);
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  cv::Mat result = fastdeploy::vision::Visualize::VisDetection(
      *(reinterpret_cast<cv::Mat*>(im)), *detection_result, score_threshold,
      line_size, font_size);
  return new cv::Mat(result);
}
}