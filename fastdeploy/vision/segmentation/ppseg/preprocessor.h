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
namespace segmentation {

class FASTDEPLOY_DECL PaddleSegPreprocessor {
 public:
  PaddleSegPreprocessor() {}
  explicit PaddleSegPreprocessor(const std::string& config_file);

  virtual bool BuildPreprocessPipelineFromConfig(
                                 const std::string& config_file);

  virtual bool Run(Mat* mat, FDTensor* outputs);

  std::vector<std::shared_ptr<Processor>> processors_;

  /** \brief For PP-HumanSeg model, set true if the input image is vertical image(height > width), default value is false
   */
  bool is_vertical_screen_ = false;

  bool is_with_softmax_ = false;

  bool is_with_argmax_ = true;

  // Paddle2ONNX temporarily don't support dynamic input
  bool is_change_backends = false;
};

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
