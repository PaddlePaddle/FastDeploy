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

#include "fastdeploy/vision/detection/ppdet/yolov3.h"

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv3::YOLOv3(const std::string& model_file, const std::string& params_file,
               const std::string& config_file,
               const RuntimeOption& custom_option,
               const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool YOLOv3::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  outputs->resize(3);
  (*outputs)[0].Allocate({1, 2}, FDDataType::FP32, "im_shape");
  (*outputs)[2].Allocate({1, 2}, FDDataType::FP32, "scale_factor");
  float* ptr0 = static_cast<float*>((*outputs)[0].MutableData());
  ptr0[0] = mat->Height();
  ptr0[1] = mat->Width();
  float* ptr2 = static_cast<float*>((*outputs)[2].MutableData());
  ptr2[0] = mat->Height() * 1.0 / origin_h;
  ptr2[1] = mat->Width() * 1.0 / origin_w;
  (*outputs)[1].name = "image";
  mat->ShareWithTensor(&((*outputs)[1]));
  // reshape to [1, c, h, w]
  (*outputs)[1].shape.insert((*outputs)[1].shape.begin(), 1);
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
