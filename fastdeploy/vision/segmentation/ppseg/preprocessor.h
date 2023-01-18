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
namespace segmentation {
/*! @brief Preprocessor object for PaddleSeg serials model.
  */
class FASTDEPLOY_DECL PaddleSegPreprocessor {
 public:
  /** \brief Create a preprocessor instance for PaddleSeg serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g ppliteseg/deploy.yaml
   */
  explicit PaddleSegPreprocessor(const std::string& config_file);

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \param[in] imgs_info The original input images shape info map, key is "shape_info", value is vector<array<int, 2>> a{{height, width}} 
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Run(
    std::vector<FDMat>* images,
    std::vector<FDTensor>* outputs,
    std::map<std::string, std::vector<std::array<int, 2>>>* imgs_info);

  /// Get is_vertical_screen property of PP-HumanSeg model, default is false
  bool GetIsVerticalScreen() const {
    return is_vertical_screen_;
  }

  /// Set is_vertical_screen value, bool type required
  void SetIsVerticalScreen(bool value) {
    is_vertical_screen_ = value;
  }

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize();
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute();

 private:
  virtual bool BuildPreprocessPipelineFromConfig();
  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;

  /** \brief For PP-HumanSeg model, set true if the input image is vertical image(height > width), default value is false
   */
  bool is_vertical_screen_ = false;

  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;

  bool is_contain_resize_op_ = false;

  bool initialized_ = false;
};

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
