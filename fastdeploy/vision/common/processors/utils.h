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

#include "fastdeploy/utils/utils.h"
#include "opencv2/core/core.hpp"
#include "fastdeploy/core/fd_tensor.h"

#ifdef ENABLE_FLYCV
#include "flycv.h" // NOLINT
#endif

namespace fastdeploy {
namespace vision {

// Convert data type of opencv to FDDataType
FDDataType OpenCVDataTypeToFD(int type);

#ifdef ENABLE_FLYCV
// Convert data type of flycv to FDDataType
FDDataType FalconCVDataTypeToFD(fcv::FCVImageType type);
// Create data type of flycv by FDDataType
fcv::FCVImageType CreateFalconCVDataType(FDDataType type, int channel = 1);
// Convert cv::Mat to fcv::Mat
fcv::Mat ConvertOpenCVMatToFalconCV(cv::Mat& im);
#endif

}  // namespace vision
}  // namespace fastdeploy
