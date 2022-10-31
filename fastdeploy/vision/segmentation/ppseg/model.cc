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

#include "fastdeploy/vision/segmentation/ppseg/model.h"

namespace fastdeploy {
namespace vision {
namespace segmentation {

PaddleSegModel::PaddleSegModel(const std::string& model_file,
                               const std::string& params_file,
                               const std::string& config_file,
                               const RuntimeOption& custom_option,
                               const ModelFormat& model_format) {
  valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::ORT, Backend::LITE};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
  preprocessor = PaddleSegPreprocessor(config_file);
}

bool PaddleSegModel::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool PaddleSegModel::Predict(cv::Mat* im, SegmentationResult* result) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data(1);

  std::map<std::string, std::array<int, 2>> im_info;
  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<int>(mat.Height()),
                            static_cast<int>(mat.Width())};
  (processed_data[0]).name = InputInfoOfRuntime(0).name;
  if (!preprocessor.Run(&mat, &(processed_data[0]))) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  if (preprocessor.is_change_backends) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
  }
  std::vector<FDTensor> infer_result(1);
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  postprocessor.is_with_argmax_ = preprocessor.is_with_argmax_;
  postprocessor.is_with_softmax_ = preprocessor.is_with_softmax_;
  if (!postprocessor.Run(&infer_result[0], result, im_info)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
