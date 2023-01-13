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
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace detection {
class FASTDEPLOY_DECL PPDetDecode {
 public:
  PPDetDecode() {}
  explicit PPDetDecode(const std::string& config_file);
  bool DecodeAndNMS(const std::vector<FDTensor>& tensors,
                    std::vector<DetectionResult>* results);

 private:
  std::string config_file_;
  std::string arch_;
  std::vector<int> fpn_stride_{8, 16, 32, 64};
  std::vector<float> im_shape_{416, 416};
  float score_threshold_ = 0.5;
  float nms_threshold_ = 0.5;
  int reg_max_ = 8;
  int num_class_ = 80;
  int batchs_ = 1;
  bool ReadPostprocessConfigFromYaml();
  void DisPred2Bbox(const float*& dfl_det, int label, float score, int x, int y,
                    int stride, fastdeploy::vision::DetectionResult* results);
  bool PicoDetPostProcess(const std::vector<FDTensor>& outs,
                          std::vector<DetectionResult>* results);
  int ActivationFunctionSoftmax(const float* src, float* dst);
};
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
