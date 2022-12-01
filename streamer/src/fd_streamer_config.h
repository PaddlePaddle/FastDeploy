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

#include "fastdeploy/utils/utils.h"


namespace fastdeploy {
namespace streamer {

struct FASTDEPLOY_DECL FDStreamerConfig {
 public:
  /** \brief Init config from YAML file
   *
   * \param[in] config_file config file path
   * \return true if the init is successful, otherwise false
   */
  bool Init(const std::string& config_file);

#ifdef ENABLE_DEEPSTREAM
  std::vector<NvUriSrcBinConfig> nv_uri_src_configs;

}
}  // namespace streamer
}  // namespace fastdeploy
