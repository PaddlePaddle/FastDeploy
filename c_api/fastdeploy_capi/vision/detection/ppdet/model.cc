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

#include "fastdeploy_capi/vision/detection/ppdet/model.h"

#include "fastdeploy_capi/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

extern "C" {

FD_C_PPYOLOEWrapper* FD_C_CreatesPPYOLOEWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper = new FD_C_PPYOLOEWrapper();
  fd_c_ppyoloe_wrapper->ppyoloe_model =
      std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>(
          new fastdeploy::vision::detection::PPYOLOE(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_ppyoloe_wrapper;
}

void FD_C_DestroyPPYOLOEWrapper(
    __fd_take FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper) {
  delete fd_c_ppyoloe_wrapper;
}

FD_C_Bool FD_C_PPYOLOEWrapperPredict(
    FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper, FD_C_Mat img,
    FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& ppyoloe_model =
      CHECK_AND_CONVERT_FD_TYPE(PPYOLOEWrapper, fd_c_ppyoloe_wrapper);
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  return ppyoloe_model->Predict(im, detection_result.get());
}
}