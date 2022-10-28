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

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace utils {

bool ChangeValidBackends(const std::string& config_file, 
                         std::vector<Backend>* valid_cpu_backends,
                         std::vector<Backend>* valid_gpu_backends) {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }   
  if (cfg["Deploy"]["input_shape"]) {
    auto input_shape = cfg["Deploy"]["input_shape"];
    int input_batch = input_shape[0].as<int>();
    int input_channel = input_shape[1].as<int>();
    int input_height = input_shape[2].as<int>();
    int input_width = input_shape[3].as<int>();
    if (input_height == -1 || input_width == -1) {
      *valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::LITE};
      *valid_gpu_backends = {Backend::PDINFER};
    }
  }
  return true;
}

}  // namespace utils
}  // namespace vision
}  // namespace fastdeploy
