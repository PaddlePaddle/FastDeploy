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
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {

namespace vision {

namespace matting {

class FASTDEPLOY_DECL MODNet : public FastDeployModel {
 public:
  MODNet(const std::string& model_file, const std::string& params_file = "",
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "matting/MODNet"; }

  // tuple of (width, height), default (256, 256)
  std::vector<int> size;
  std::vector<float> alpha;
  std::vector<float> beta;
  // whether to swap the B and R channel, such as BGR->RGB, default true.
  bool swap_rb;

  bool Predict(cv::Mat* im, MattingResult* result);

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* output,
                  std::map<std::string, std::array<int, 2>>* im_info);

  bool Postprocess(std::vector<FDTensor>& infer_result, MattingResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);
};

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy
