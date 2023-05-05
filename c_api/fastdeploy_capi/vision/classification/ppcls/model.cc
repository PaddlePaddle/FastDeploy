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

#include "fastdeploy_capi/internal/types_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

FD_C_PaddleClasModelWrapper* FD_C_CreatePaddleClasModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper =
      new FD_C_PaddleClasModelWrapper();
  fd_c_paddleclas_model_wrapper->paddleclas_model =
      std::unique_ptr<fastdeploy::vision::classification::PaddleClasModel>(
          new fastdeploy::vision::classification::PaddleClasModel(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_paddleclas_model_wrapper;
}

void FD_C_DestroyPaddleClasModelWrapper(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper) {
  delete fd_c_paddleclas_model_wrapper;
}

FD_C_Bool FD_C_PaddleClasModelWrapperPredict(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper, FD_C_Mat img,
    FD_C_ClassifyResult* fd_c_classify_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_c_paddleclas_model_wrapper);
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      FD_C_CreateClassifyResultWrapper();
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);

  bool successful = paddleclas_model->Predict(im, classify_result.get());
  if (successful) {
    FD_C_ClassifyResultWrapperToCResult(fd_c_classify_result_wrapper,
                                        fd_c_classify_result);
  }
  FD_C_DestroyClassifyResultWrapper(fd_c_classify_result_wrapper);
  return successful;
}

FD_C_Bool FD_C_PaddleClasModelWrapperInitialized(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper) {
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_c_paddleclas_model_wrapper);
  return paddleclas_model->Initialized();
}

FD_C_Bool FD_C_PaddleClasModelWrapperBatchPredict(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper,
    FD_C_OneDimMat imgs, FD_C_OneDimClassifyResult* results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<FD_C_ClassifyResultWrapper*> results_wrapper_out;
  std::vector<fastdeploy::vision::ClassifyResult> results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
    FD_C_ClassifyResultWrapper* fd_classify_result_wrapper =
        FD_C_CreateClassifyResultWrapper();
    results_wrapper_out.push_back(fd_classify_result_wrapper);
  }
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_c_paddleclas_model_wrapper);
  bool successful = paddleclas_model->BatchPredict(imgs_vec, &results_out);
  if (successful) {
    // copy results back to FD_C_OneDimClassifyResult
    results->size = results_out.size();
    results->data = new FD_C_ClassifyResult[results->size];
    for (int i = 0; i < results_out.size(); i++) {
      (*CHECK_AND_CONVERT_FD_TYPE(ClassifyResultWrapper,
                                  results_wrapper_out[i])) =
          std::move(results_out[i]);
      FD_C_ClassifyResultWrapperToCResult(results_wrapper_out[i],
                                          &results->data[i]);
    }
  }
  for (int i = 0; i < results_out.size(); i++) {
    FD_C_DestroyClassifyResultWrapper(results_wrapper_out[i]);
  }
  return successful;
}

#ifdef __cplusplus
}
#endif
