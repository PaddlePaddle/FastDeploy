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
#include "fastdeploy/vision/common/processors/normalize.h"
#include "fastdeploy/vision/common/processors/stride_pad.h"
#include "fastdeploy/vision/common/processors/resize.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace classification {
/*! @brief Preprocessor object for DBDetector serials model.
 */
class FASTDEPLOY_DECL ObjDetectorPreprocessor : public ProcessorManager {
 public:
  ObjDetectorPreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(FDMatBatch* image_batch, std::vector<FDTensor>* outputs);

  void Init(const YAML::Node& config_node) {
    for (int i = 0; i < config_node.size(); ++i) {
      if (config_node[i]["DetResize"].IsDefined()) {
        int interp = config_node[i]["DetResize"]["interp"].as<int>();
        bool keep_ratio_ = item["keep_ratio"].as<bool>();
        std::vector<int> target_size_ = item["target_size"].as < std::vector < int >> ();
        ops_["Resize"] = std::make_shared<Resize>(target_size_[0], target_size_[1], , , interp, keep_ratio_);
      }

      if (config_node[i]["DetNormalizeImage"].IsDefined()) {
        std::vector < float > mean_ = item["mean"].as < std::vector < float >> ();
        std::vector < float > scale_ = item["std"].as < std::vector < float >> ();
        bool is_scale_ = item["is_scale"].as<bool>();
        ops_["NormalizeImage"] = std::make_shared<Normalize>(mean_, , is_scale_, , , false);
      }

      if (config_node[i]["DetPermute"].IsDefined()) {
        ops_["Permute"] = std::make_shared<HWC2CHW>();
      }

      if (config_node[i]["DetPadStrid"].IsDefined()) {
        ops_["PadStride"] = std::make_shared<StridePad>();
      }
    }
  }

 private:
  std::unordered_map <std::string, std::shared_ptr<Processor>> ops_;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
