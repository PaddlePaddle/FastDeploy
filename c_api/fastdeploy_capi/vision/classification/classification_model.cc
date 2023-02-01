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

#include "fastdeploy_capi/vision/classification/classification_model.h"

#include "fastdeploy_capi/types_internal.h"

namespace fastdeploy {
std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>& CheckAndConvertFD_PaddleClasModel(FD_PaddleClasModel* fd_paddleclas_model){
  FDASSERT(fd_paddleclas_model != nullptr,
           "The pointer of fd_paddleclas_model shouldn't be nullptr.");
  return fd_paddleclas_model->paddleclas_model;
}
}  // namespace fastdeploy

extern "C" {

FD_PaddleClasModel* FD_CreatePaddleClasModel(
    const char* model_file, const char* params_file, const char* config_file,
    FD_RuntimeOption* fd_runtime_option, const FD_ModelFormat model_format) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  FD_PaddleClasModel* fd_paddleclas_model = new FD_PaddleClasModel();
  fd_paddleclas_model->paddleclas_model =
      std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>(
          new fastdeploy::vision::classification::PaddleClasModel(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_paddleclas_model;
}

void FD_DestroyPaddleClasModel(
    __fd_take FD_PaddleClasModel* fd_paddleclas_model) {
  delete fd_paddleclas_model;
}

void FD_PaddleClasModelPredict(
    __fd_take FD_PaddleClasModel* fd_paddleclas_model, FD_Mat* img,
    FD_ClassifyResult* fd_classify_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  CHECK_AND_CONVERT_FD_PaddleClasModel;
  CHECK_AND_CONVERT_FD_ClassifyResult;
  paddleclas_model->Predict(im, classify_result.get());
}
}