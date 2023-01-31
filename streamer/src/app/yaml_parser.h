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

#include "yaml-cpp/yaml.h"
#include "app/base_app.h"
#include <gst/gst.h>

namespace fastdeploy {
namespace streamer {

/*! @brief YAML Parser class, to parse stream configs from yaml file
 */
class YamlParser {
 public:
  /** \brief Create a YAML parser
   *
   * \param[in] config_file Path of configuration file
   */
  explicit YamlParser(const std::string& config_file);

  void ParseAppConfg(AppConfig& app_config);

  void ValidateConfig();

  GstElement* BuildPipelineFromConfig();

 private:
  std::string YamlToPipelineDescStr();

  void ParseElement(const std::string& name, const YAML::Node& properties);

  void ParseNvUriSrcBinList(const std::string& name,
                            const YAML::Node& properties);

  static std::string ParseProperty(const YAML::Node& name,
                                   const YAML::Node& value);

  AppConfig app_config_;
  std::string config_file_;
  YAML::Node yaml_config_;
  GstElement* pipeline_;
  std::vector<std::string> elem_descs_;
};
}  // namespace streamer
}  // namespace fastdeploy
