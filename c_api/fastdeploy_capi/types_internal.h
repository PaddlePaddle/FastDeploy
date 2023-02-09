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

#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy_capi/fd_type.h"
#include <memory>

#ifdef ENABLE_VISION
#include "fastdeploy/vision/classification/ppcls/model.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/detection/ppdet/model.h"

typedef struct FD_C_ClassifyResultWrapper {
  std::unique_ptr<fastdeploy::vision::ClassifyResult> classify_result;
} FD_C_ClassifyResultWrapper;

typedef struct FD_C_DetectionResultWrapper {
  std::unique_ptr<fastdeploy::vision::DetectionResult> detection_result;
} FD_C_DetectionResultWrapper;

typedef struct FD_C_PaddleClasModelWrapper {
  std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>
      paddleclas_model;
} FD_C_PaddleClasModelWrapper;

typedef struct FD_C_PPYOLOEWrapper {
  std::unique_ptr<fastdeploy::vision::detection::PPYOLOE> ppyoloe_model;
} FD_C_PPYOLOEWrapper;

namespace fastdeploy {
std::unique_ptr<fastdeploy::vision::ClassifyResult>&
FD_C_CheckAndConvertClassifyResultWrapper(
    FD_C_ClassifyResultWrapper* fd_classify_result_wrapper);
std::unique_ptr<fastdeploy::vision::DetectionResult>&
FD_C_CheckAndConvertDetectionResultWrapper(
    FD_C_DetectionResultWrapper* fd_detection_result_wrapper);
std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>&
FD_C_CheckAndConvertPaddleClasModelWrapper(
    FD_C_PaddleClasModelWrapper* fd_paddleclas_model_wrapper);
std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>&
FD_C_CheckAndConvertPPYOLOEWrapper(FD_C_PPYOLOEWrapper* fd_ppyoloe_wrapper);
}  // namespace fastdeploy

#endif

typedef struct FD_C_RuntimeOptionWrapper {
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_option;
} FD_C_RuntimeOptionWrapper;

namespace fastdeploy {
std::unique_ptr<fastdeploy::RuntimeOption>&
FD_C_CheckAndConvertRuntimeOptionWrapper(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper);
}

#define CHECK_AND_CONVERT_FD_TYPE(TYPENAME, variable_name)                     \
  fastdeploy::FD_C_CheckAndConvert##TYPENAME(variable_name)
