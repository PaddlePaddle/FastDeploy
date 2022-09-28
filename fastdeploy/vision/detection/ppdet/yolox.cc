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

#include "fastdeploy/vision/detection/ppdet/yolox.h"

namespace fastdeploy {
namespace vision {
namespace detection {

PaddleYOLOX::PaddleYOLOX(const std::string& model_file,
                         const std::string& params_file,
                         const std::string& config_file,
                         const RuntimeOption& custom_option,
                         const ModelFormat& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  background_label = -1;
  keep_top_k = 1000;
  nms_eta = 1;
  nms_threshold = 0.65;
  nms_top_k = 10000;
  normalized = true;
  score_threshold = 0.001;
  initialized = Initialize();
}

bool PaddleYOLOX::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  float scale[2] = {1.0, 1.0};
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
    if (processors_[i]->Name().find("Resize") != std::string::npos) {
      scale[0] = mat->Height() * 1.0 / origin_h;
      scale[1] = mat->Width() * 1.0 / origin_w;
    }
  }

  outputs->resize(2);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));

  // reshape to [1, c, h, w]
  (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);

  (*outputs)[1].Allocate({1, 2}, FDDataType::FP32, InputInfoOfRuntime(1).name);
  float* ptr = static_cast<float*>((*outputs)[1].MutableData());
  ptr[0] = scale[0];
  ptr[1] = scale[1];
  return true;
}
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
