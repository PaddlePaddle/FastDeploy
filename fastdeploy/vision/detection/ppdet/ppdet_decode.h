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
#include "fastdeploy/vision/detection/ppdet/multiclass_nms.h"

namespace fastdeploy {
namespace vision {
namespace detection {
class FASTDEPLOY_DECL PPDetDecode {
 public:
  PPDetDecode() = default;
  explicit PPDetDecode(const std::string& config_file);
  bool DecodeAndNMS(const std::vector<FDTensor>& tensors,
                    std::vector<DetectionResult>* results);
  void SetNMSOption(const NMSOption& option = NMSOption()) {
    multi_class_nms_.SetNMSOption(option);
  }

 private:
  std::string config_file_;
  std::string arch_;
  std::vector<int> fpn_stride_{8, 16, 32, 64};
  std::vector<float> im_shape_{416, 416};
  int batchs_ = 1;
  bool ReadPostprocessConfigFromYaml();
  void DisPred2Bbox(const float*& dfl_det, int label, float score, int x, int y,
                    int stride, fastdeploy::vision::DetectionResult* results,
                    int reg_max, int num_class);
  bool PicoDetPostProcess(const std::vector<FDTensor>& outs,
                          std::vector<DetectionResult>* results, int reg_max,
                          int num_class);
  int ActivationFunctionSoftmax(const float* src, float* dst, int reg_max);
  PaddleMultiClassNMS multi_class_nms_;
};
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
