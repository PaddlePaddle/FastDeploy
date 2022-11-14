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
#include "fastdeploy/vision/sr/ppsr/ppmsvsr.h"

namespace fastdeploy {
namespace vision {
namespace sr {


class FASTDEPLOY_DECL EDVR : public PPMSVSR{
 public:
  EDVR(const std::string& model_file,
       const std::string& params_file,
       const RuntimeOption& custom_option = RuntimeOption(),
       const ModelFormat& model_format = ModelFormat::PADDLE);
  /// model name contained EDVR
  std::string ModelName() const override { return "EDVR"; }

 private:
  bool Postprocess(std::vector<FDTensor>& infer_results,
                   std::vector<cv::Mat>& results) override;
};
}  // namespace sr
}  // namespace vision
}  // namespace fastdeploy
