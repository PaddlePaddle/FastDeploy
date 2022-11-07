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

#include "fastdeploy/vision/classification/ppcls/model.h"

namespace fastdeploy {
namespace vision {
namespace classification {

PaddleClasModel::PaddleClasModel(const std::string& model_file,
                                 const std::string& params_file,
                                 const std::string& config_file,
                                 const RuntimeOption& custom_option,
                                 const ModelFormat& model_format) : preprocessor_(config_file) {
  valid_cpu_backends = {Backend::ORT, Backend::OPENVINO, Backend::PDINFER,
                        Backend::LITE};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PaddleClasModel::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool PaddleClasModel::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
  postprocessor_.SetTopk(topk);
  if (!Predict(*im, result)) {
    return false;
  }
  return true;
}

bool PaddleClasModel::Predict(const cv::Mat& im, ClassifyResult* result) {
  std::vector<ClassifyResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool PaddleClasModel::BatchPredict(const std::vector<cv::Mat>& images, std::vector<ClassifyResult>* results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors, &reused_output_tensors)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime." << std::endl;
    return false;
  }

  return true;
}

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
