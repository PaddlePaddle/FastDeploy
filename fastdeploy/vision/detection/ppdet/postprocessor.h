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
/*! @brief Postprocessor object for PaddleDet serials model.
 */
class FASTDEPLOY_DECL PaddleDetPostprocessor {
 public:
  PaddleDetPostprocessor() = default;
  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor>& tensors,
          std::vector<DetectionResult>* result);
  /// This function will enable decode and nms in postprocess step.
  void ApplyDecodeAndNMS();

 private:
  // Process mask tensor for MaskRCNN
  bool ProcessMask(const FDTensor& tensor,
            std::vector<DetectionResult>* results);


  bool apply_decode_and_nms_ = false;
  bool ProcessUnDecodeResults(const std::vector<FDTensor>& tensors,
                              std::vector<DetectionResult>* results);
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
