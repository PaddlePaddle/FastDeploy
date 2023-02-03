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

#include <memory>
#include "fastdeploy_capi/fd_type.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/vision/classification/ppcls/model.h"
#include "fastdeploy/vision/detection/ppdet/model.h"



typedef struct FD_RuntimeOption {
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_option;
} FD_RuntimeOption;

typedef struct FD_ClassifyResultData{
  FD_OneDimArrayInt32 label_ids;
  FD_OneDimArrayFloat scores;
  fastdeploy::vision::ResultType type;
} FD_ClassifyResultData;

typedef struct FD_ClassifyResult {
  std::unique_ptr<fastdeploy::vision::ClassifyResult> classify_result;
} FD_ClassifyResult;

typedef struct FD_DetectionResult {
  std::unique_ptr<fastdeploy::vision::DetectionResult> detection_result;
} FD_DetectionResult;

typedef struct FD_PaddleClasModel {
  std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel> paddleclas_model;
} FD_PaddleClasModel;

typedef struct FD_PPYOLOE {
  std::unique_ptr<fastdeploy::vision::detection::PPYOLOE> ppyoloe_model;
} FD_PPYOLOE;


namespace fastdeploy{
std::unique_ptr<fastdeploy::RuntimeOption>& CheckAndConvertFD_RuntimeOption(FD_RuntimeOption* fd_runtime_option);
std::unique_ptr<fastdeploy::vision::ClassifyResult>& CheckAndConvertFD_ClassifyResult(FD_ClassifyResult* fd_classify_result);
std::unique_ptr<fastdeploy::vision::DetectionResult>& CheckAndConvertFD_DetectionResult(FD_DetectionResult* fd_detection_result);
std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>& CheckAndConvertFD_PaddleClasModel(FD_PaddleClasModel* fd_paddleclas_model);
std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>& CheckAndConvertFD_PPYOLOE(FD_PPYOLOE* fd_ppyoloe_model);
}
#define CHECK_AND_CONVERT_FD_RuntimeOption  auto& runtime_option = fastdeploy::CheckAndConvertFD_RuntimeOption(fd_runtime_option)

#define CHECK_AND_CONVERT_FD_PaddleClasModel  auto& paddleclas_model = fastdeploy::CheckAndConvertFD_PaddleClasModel(fd_paddleclas_model)
#define CHECK_AND_CONVERT_FD_PPYOLOE  auto& ppyoloe_model = fastdeploy::CheckAndConvertFD_PPYOLOE(fd_ppyoloe_model)

#define CHECK_AND_CONVERT_FD_ClassifyResult  auto& classify_result = fastdeploy::CheckAndConvertFD_ClassifyResult(fd_classify_result)
#define CHECK_AND_CONVERT_FD_DetectionResult  auto& detection_result = fastdeploy::CheckAndConvertFD_DetectionResult(fd_detection_result)
