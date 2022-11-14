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
/*! @brief Preprocessor object for PaddleDet serials model.
 */
class FASTDEPLOY_DECL PaddleDetPreprocessor {
 public:
  PaddleDetPreprocessor() = default;
  /** \brief Create a preprocessor instance for PaddleDet serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
   */
  explicit PaddleDetPreprocessor(const std::string& config_file);

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime, include image, scale_factor, im_shape
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs);

 private:
  bool BuildPreprocessPipelineFromConfig(const std::string& config_file);
  std::vector<std::shared_ptr<Processor>> processors_;
  bool initialized_ = false;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
