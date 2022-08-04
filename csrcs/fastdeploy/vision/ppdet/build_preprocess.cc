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

#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/ppdet/ppyoloe.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {

bool BuildPreprocessPipelineFromConfig(
    std::vector<std::shared_ptr<Processor>>* processors,
    const std::string& config_file) {
  processors->clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  processors->push_back(std::make_shared<BGR2RGB>());

  for (const auto& op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = op["is_scale"].as<bool>();
      processors->push_back(std::make_shared<Normalize>(mean, std, is_scale));
    } else if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size(),
               "Require size of target_size be 2, but now it's " +
                   std::to_string(target_size.size()) + ".");
      if (!keep_ratio) {
        int width = target_size[1];
        int height = target_size[0];
        processors->push_back(
            std::make_shared<Resize>(width, height, -1.0, -1.0, interp, false));
      } else {
        int min_target_size = std::min(target_size[0], target_size[1]);
        int max_target_size = std::max(target_size[0], target_size[1]);
        processors->push_back(std::make_shared<ResizeByShort>(
            min_target_size, interp, true, max_target_size));
      }
    } else if (op_name == "Permute") {
      // Do nothing, do permute as the last operation
      continue;
    } else if (op_name == "Pad") {
      auto size = op["size"].as<std::vector<int>>();
      auto value = op["fill_value"].as<std::vector<float>>();
      processors->push_back(std::make_shared<Cast>("float"));
      processors->push_back(
          std::make_shared<PadToSize>(size[1], size[0], value));
    } else if (op_name == "PadStride") {
      auto stride = op["stride"].as<int>();
      processors->push_back(
          std::make_shared<StridePad>(stride, std::vector<float>(3, 0)));
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  processors->push_back(std::make_shared<HWC2CHW>());
  return true;
}

}  // namespace vision
}  // namespace fastdeploy
