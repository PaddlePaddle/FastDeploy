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

#include "fastdeploy/vision/ocr/ppocr/dbdetector.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

DBDetector::DBDetector() {}
DBDetector::DBDetector(const std::string& model_file,
                       const std::string& params_file,
                       const RuntimeOption& custom_option,
                       const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT,
                          Backend::OPENVINO};  
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

// Init
bool DBDetector::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool DBDetector::Predict(const cv::Mat& img,
                         std::vector<std::array<int, 8>>* boxes_result) {
  std::vector<std::vector<std::array<int, 8>>> det_results;
  if (!BatchPredict({img}, &det_results)) {
    return false;
  }
  *boxes_result = std::move(det_results[0]);
  return true;
}

bool DBDetector::BatchPredict(const std::vector<cv::Mat>& images,
                              std::vector<std::vector<std::array<int, 8>>>* det_results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  std::vector<std::array<int, 4>> batch_det_img_info;
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_, &batch_det_img_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, det_results, batch_det_img_info)) {
    FDERROR << "Failed to postprocess the inference cls_results by runtime." << std::endl;
    return false;
  }
  return true;
}

}  // namesapce ocr
}  // namespace vision
}  // namespace fastdeploy
