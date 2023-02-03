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

namespace fastdeploy {
std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>&
CheckAndConvertFD_PPYOLOE(FD_PPYOLOE* fd_ppyoloe_model) {
  FDASSERT(fd_ppyoloe_model != nullptr,
           "The pointer of fd_paddleclas_model shouldn't be nullptr.");
  return fd_ppyoloe_model->ppyoloe_model;
}
}  // namespace fastdeploy

extern "C" {

FD_PPYOLOE* FD_CreatesPPYOLOE(const char* model_file, const char* params_file,
                              const char* config_file,
                              FD_RuntimeOption* fd_runtime_option,
                              const FD_ModelFormat model_format) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  FD_PPYOLOE* fd_ppyoloe_model = new FD_PPYOLOE();
  fd_ppyoloe_model->ppyoloe_model =
      std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>(
          new fastdeploy::vision::detection::PPYOLOE(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_ppyoloe_model;
}

void FD_DestroyPPYOLOE(__fd_take FD_PPYOLOE* fd_ppyoloe_model) {
  delete fd_ppyoloe_model;
}

void FD_PPYOLOEPredict(FD_PPYOLOE* fd_ppyoloe_model, FD_Mat* img,
                       FD_DetectionResult* fd_detection_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  CHECK_AND_CONVERT_FD_PPYOLOE;
  CHECK_AND_CONVERT_FD_DetectionResult;
  ppyoloe_model->Predict(im, detection_result.get());
}
}