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

#include "fastdeploy_capi/fd_common.h"
#include "fastdeploy_capi/fd_type.h"
#include "fastdeploy_capi/vision/result.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Visualize Detection
 *
 * \return Return a pointer to cv::Mat object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat
FD_C_VisDetection(FD_C_Mat im, FD_C_DetectionResult* fd_detection_result,
                  float score_threshold, int line_size, float font_size);

#ifdef __cplusplus
}  // extern "C"
#endif
