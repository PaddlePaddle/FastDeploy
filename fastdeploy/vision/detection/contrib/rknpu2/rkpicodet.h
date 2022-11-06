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
class FASTDEPLOY_DECL RKPicoDet : public PPYOLOE {
 public:
  RKPicoDet(const std::string& model_file,
          const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::RKNN);

  virtual std::string ModelName() const { return "RKPicoDet"; }

 protected:
  /// Build the preprocess pipeline from the loaded model
  virtual bool BuildPreprocessPipelineFromConfig();
  /// Preprocess an input image, and set the preprocessed results to `outputs`
  virtual bool Preprocess(Mat* mat, std::vector<FDTensor>* outputs);

  /// Postprocess the inferenced results, and set the final result to `result`
  virtual bool Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result);
  virtual bool Initialize();
 private:
  std::vector<float> scale_factor{1.0, 1.0};
};
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
