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
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace segmentation {

class FASTDEPLOY_DECL PaddleSegPostprocessor {
 public:
  PaddleSegPostprocessor() {}

  virtual bool Run(FDTensor* infer_result,
                   SegmentationResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);

  /** \brief Whether applying softmax operator in the postprocess, default value is false
   */
  bool apply_softmax_ = false;

  bool is_with_softmax_ = false;

  bool is_with_argmax_ = true;
};

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
