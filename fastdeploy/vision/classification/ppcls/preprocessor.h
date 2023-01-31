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
#include "fastdeploy/vision/common/processors/manager.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace classification {
/*! @brief Preprocessor object for PaddleClas serials model.
 */
class FASTDEPLOY_DECL PaddleClasPreprocessor : public ProcessorManager {
 public:
  /** \brief Create a preprocessor instance for PaddleClas serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
   */
  explicit PaddleClasPreprocessor(const std::string& config_file);

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(std::vector<FDMat>* images,
                     std::vector<FDTensor>* outputs);

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize();
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute();

 private:
  bool BuildPreprocessPipelineFromConfig();
  std::vector<std::shared_ptr<Processor>> processors_;
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  // read config file
  std::string config_file_;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
