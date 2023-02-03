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

FD_Mat* FD_VisDetection(FD_Mat* im, FD_DetectionResult* fd_detection_result,
                        int line_size, float font_size) {
  CHECK_AND_CONVERT_FD_DetectionResult;
  cv::Mat result = fastdeploy::vision::Visualize::VisDetection(
      *(reinterpret_cast<cv::Mat*>(im)), *detection_result, line_size,
      font_size);
  return new cv::Mat(result);
}
}