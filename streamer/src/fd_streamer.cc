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

#include "fd_streamer.h"

#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace streamer {

bool FDStreamer::Init(const std::string& config_file) {
  YAML::Node cfg;
  std::cout << "haha" << std::endl;
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }
  std::cout << "hehe" << std::endl;
  // auto preprocess_cfg = cfg;
  for (const auto& op : cfg) {
    // FDASSERT(op.IsMap(),
            //  "Require the transform information in yaml be Map type.");
    auto op_name = op.as<std::string>();
    std::cout << op_name << std::endl;
  }
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
