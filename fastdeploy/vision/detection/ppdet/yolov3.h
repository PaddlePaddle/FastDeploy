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

#pragma once
#include "fastdeploy/vision/detection/ppdet/ppyoloe.h"

namespace fastdeploy {
namespace vision {
namespace detection {

class FASTDEPLOY_DECL YOLOv3 : public PPYOLOE {
 public:
  YOLOv3(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::PADDLE);

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv3"; }

  virtual bool Preprocess(Mat* mat, std::vector<FDTensor>* outputs);
};
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
