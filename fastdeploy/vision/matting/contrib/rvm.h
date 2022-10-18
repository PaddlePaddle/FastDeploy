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

class FASTDEPLOY_DECL RobustVideoMatting : public FastDeployModel {
 public:
  RobustVideoMatting(const std::string& model_file,
                     const std::string& params_file = "",
                     const RuntimeOption& custom_option = RuntimeOption(),
                     const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "matting/RobustVideoMatting"; }

  // tuple of (width, height), default (1080, 1920)
  std::vector<int> size;

  bool Predict(cv::Mat* im, MattingResult* result);

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* output,
                  std::map<std::string, std::array<int, 2>>* im_info);

  bool Postprocess(std::vector<FDTensor>& infer_result, MattingResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);
  // init dynamic inputs datas
  std::vector<std::vector<float>> dynamic_inputs_datas_ = {
     {0.0f},  // r1i
     {0.0f},  // r2i
     {0.0f},  // r3i
     {0.0f},  // r4i
     {0.25f},  // downsample_ratio
  };
  // init dynamic inputs dims
  std::vector<std::vector<int64_t>> dynamic_inputs_dims_ = {
     {1, 1, 1, 1},  // r1i
     {1, 1, 1, 1},  // r2i
     {1, 1, 1, 1},  // r3i
     {1, 1, 1, 1},  // r4i
     {1},  // downsample_ratio
  };
};

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy
