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
#include "fastdeploy/core/fd_tensor.h"

namespace fastdeploy {
namespace vision {

// Convert data type of opencv to FDDataType
FDDataType OpenCVDataTypeToFD(int type);

#ifdef ENABLE_FALCONCV
// Convert data type of falconcv to FDDataType
FDDataType FalconCVDataTypeToFD(fcv::FCVImageType type);
// Create data type of falconcv by FDDataType
fcv::FCVImageType CreateFalconCVDataType(FDDataType type, int channel = 1);
#endif

} // namespace vision
} // namespace fastdeploy
