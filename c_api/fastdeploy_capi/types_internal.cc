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

#include "fastdeploy_capi/types_internal.h"

namespace fastdeploy {

#ifdef ENABLE_VISION

std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>&
FD_C_CheckAndConvertPaddleClasModelWrapper(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper) {
  FDASSERT(
      fd_c_paddleclas_model_wrapper != nullptr,
      "The pointer of fd_c_paddleclas_model_wrapper shouldn't be nullptr.");
  return fd_c_paddleclas_model_wrapper->paddleclas_model;
}

std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>&
FD_C_CheckAndConvertPPYOLOEWrapper(FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper) {
  FDASSERT(fd_c_ppyoloe_wrapper != nullptr,
           "The pointer of fd_c_ppyoloe_wrapper shouldn't be nullptr.");
  return fd_c_ppyoloe_wrapper->ppyoloe_model;
}

std::unique_ptr<fastdeploy::vision::ClassifyResult>&
FD_C_CheckAndConvertClassifyResultWrapper(
    FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper) {
  FDASSERT(fd_c_classify_result_wrapper != nullptr,
           "The pointer of fd_c_classify_result_wrapper shouldn't be nullptr.");
  return fd_c_classify_result_wrapper->classify_result;
}

std::unique_ptr<fastdeploy::vision::DetectionResult>&
FD_C_CheckAndConvertDetectionResultWrapper(
    FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  FDASSERT(
      fd_c_detection_result_wrapper != nullptr,
      "The pointer of fd_c_detection_result_wrapper shouldn't be nullptr.");
  return fd_c_detection_result_wrapper->detection_result;
}
#endif

std::unique_ptr<fastdeploy::RuntimeOption>&
FD_C_CheckAndConvertRuntimeOptionWrapper(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  FDASSERT(fd_c_runtime_option_wrapper != nullptr,
           "The pointer of fd_c_runtime_option_wrapper shouldn't be nullptr.");
  return fd_c_runtime_option_wrapper->runtime_option;
}

}  // namespace fastdeploy