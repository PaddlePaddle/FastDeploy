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
    FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_c_paddleclas_model_wrapper);
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  return paddleclas_model->Predict(im, classify_result.get());
}

FD_C_Bool FD_C_PaddleClasModelWrapperInitialized(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper) {
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_c_paddleclas_model_wrapper);
  return paddleclas_model->Initialized();
}

FD_C_ClassifyResult* FD_C_ClassifyResultToC(
    fastdeploy::vision::ClassifyResult* classify_result) {
  // Internal use, transfer fastdeploy::vision::ClassifyResult to
  // FD_C_ClassifyResult
  FD_C_ClassifyResult* fd_c_classify_result_data = new FD_C_ClassifyResult();
  // copy label_ids
  fd_c_classify_result_data->label_ids.size = classify_result->label_ids.size();
  fd_c_classify_result_data->label_ids.data =
      new int32_t[fd_c_classify_result_data->label_ids.size];
  memcpy(fd_c_classify_result_data->label_ids.data,
         classify_result->label_ids.data(),
         sizeof(int32_t) * fd_c_classify_result_data->label_ids.size);
  // copy scores
  fd_c_classify_result_data->scores.size = classify_result->scores.size();
  fd_c_classify_result_data->scores.data =
      new float[fd_c_classify_result_data->scores.size];
  memcpy(fd_c_classify_result_data->scores.data, classify_result->scores.data(),
         sizeof(float) * fd_c_classify_result_data->scores.size);
  fd_c_classify_result_data->type =
      static_cast<FD_C_ResultType>(classify_result->type);
  return fd_c_classify_result_data;
}

FD_C_Bool FD_C_PaddleClasModelWrapperBatchPredict(
    FD_C_PaddleClasModelWrapper* fd_c_paddleclas_model_wrapper,
    FD_C_OneDimMat imgs, FD_C_OneDimClassifyResult* results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<fastdeploy::vision::ClassifyResult> results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  auto& paddleclas_model = CHECK_AND_CONVERT_FD_TYPE(
      PaddleClasModelWrapper, fd_c_paddleclas_model_wrapper);
  bool successful = paddleclas_model->BatchPredict(imgs_vec, &results_out);
  if (successful) {
    // copy results back to FD_C_OneDimClassifyResult
    results->size = results_out.size();
    results->data = new FD_C_ClassifyResult[results->size];
    for (int i = 0; i < results_out.size(); i++) {
      results->data[i] = *FD_C_ClassifyResultToC(&results_out[i]);
    }
  }
  return successful;
}

#ifdef __cplusplus
}
#endif
