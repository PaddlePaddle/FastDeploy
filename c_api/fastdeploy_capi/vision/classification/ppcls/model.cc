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

#include "fastdeploy_capi/vision/classification/ppcls/model.h"

#include "fastdeploy_capi/types_internal.h"

namespace fastdeploy {
std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>&
CheckAndConvertFD_PaddleClasModelWrapper(
    FD_PaddleClasModelWrapper* fd_paddleclas_model_wrapper) {
  FDASSERT(fd_paddleclas_model_wrapper != nullptr,
           "The pointer of fd_paddleclas_model_wrapper shouldn't be nullptr.");
  return fd_paddleclas_model_wrapper->paddleclas_model;
}
}  // namespace fastdeploy

extern "C" {

FD_PaddleClasModelWrapper* FD_CreatePaddleClasModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_RuntimeOptionWrapper* fd_runtime_option_wrapper,
    const FD_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_runtime_option_wrapper);
  FD_PaddleClasModelWrapper* fd_paddleclas_model_wrapper =
      new FD_PaddleClasModelWrapper();
  fd_paddleclas_model_wrapper->paddleclas_model =
      std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>(
          new fastdeploy::vision::classification::PaddleClasModel(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_paddleclas_model_wrapper;
}

void FD_DestroyPaddleClasModelWrapper(
    __fd_take FD_PaddleClasModelWrapper* fd_paddleclas_model_wrapper) {
  delete fd_paddleclas_model_wrapper;
}

void FD_PaddleClasModelWrapperPredict(
    __fd_take FD_PaddleClasModelWrapper* fd_paddleclas_model_wrapper,
    FD_Mat* img, FD_ClassifyResultWrapper* fd_classify_result_wrapper) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_paddleclas_model_wrapper);
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(ClassifyResultWrapper,
                                                    fd_classify_result_wrapper);
  paddleclas_model->Predict(im, classify_result.get());
}
}