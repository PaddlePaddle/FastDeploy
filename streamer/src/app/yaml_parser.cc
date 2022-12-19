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

#include "app/yaml_parser.h"
#include "gstreamer/utils.h"

namespace fastdeploy {
namespace streamer {

YamlParser::YamlParser(const std::string& config_file) {
  try {
    yaml_config_ = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
  }
  config_file_ = config_file;
}

void YamlParser::ParseAppConfg(AppConfig& app_config) {
  ValidateConfig();
  auto elem = yaml_config_["app"];

  auto type_str = elem["type"].as<std::string>();
  if (type_str == "video_analytics") {
    app_config.type = AppType::VIDEO_ANALYTICS;
  } else if (type_str == "video_decoder") {
    app_config.type = AppType::VIDEO_DECODER;
  } else {
    FDASSERT(false, "Unsupported app type: %s.", type_str.c_str());
  }

  app_config.enable_perf_measurement = elem["enable-perf-measurement"].as<bool>();
  if (app_config.enable_perf_measurement) {
    app_config.perf_interval_sec = elem["perf-measurement-interval-sec"].as<int>();
  }
  app_config_ = app_config;
}

void YamlParser::ValidateConfig() {
  auto first_elem = yaml_config_.begin()->first.as<std::string>();
  if (first_elem != "app") {
    FDASSERT(false, "First config element must be app, but got %s.",
             first_elem.c_str());
  }
}

GstElement* YamlParser::BuildPipelineFromConfig() {
  auto pipeline_desc = YamlToPipelineDescStr();
  pipeline_ = CreatePipeline(pipeline_desc);
  return pipeline_;
}

std::string YamlParser::YamlToPipelineDescStr() {
  for (const auto& elem : yaml_config_) {
    std::string elem_name = elem.first.as<std::string>();
    std::cout << elem_name << std::endl;
    ParseElement(elem_name, elem.second);
  }
  std::string pipeline_desc = "";
  for (size_t i = 0; i < elem_descs_.size(); i++) {
    pipeline_desc += elem_descs_[i];
    if (elem_descs_[i].find('!') != std::string::npos) continue;
    if (i >= elem_descs_.size() - 1) continue;
    pipeline_desc += "! ";
  }
  return pipeline_desc;
}

void YamlParser::ParseElement(const std::string& name, const YAML::Node& properties) {
  if (name == "app") return;

  if (name == "nvurisrcbin_list") {
    ParseNvUriSrcBinList(name, properties);
    return;
  }

  std::string elem_desc = name + " ";
  for (auto it = properties.begin(); it != properties.end(); it++) {
    elem_desc += ParseProperty(it->first, it->second) + " ";
  }
  elem_descs_.push_back(elem_desc);
}

void YamlParser::ParseNvUriSrcBinList(const std::string& name, const YAML::Node& properties) {
  std::string elem_name = "nvurisrcbin";
  
  auto uri_list = properties["uri-list"].as<std::vector<std::string>>();
  auto pad_prefix = properties["pad-prefix"].as<std::string>();
  for (size_t i = 0; i < uri_list.size(); i++) {
    std::string elem_desc = elem_name + " ";
    elem_desc += "uri=" + uri_list[i] + " ";
    for (auto it = properties.begin(); it != properties.end(); it++) {
      auto prop_name = it->first.as<std::string>();
      if (prop_name == "uri-list" || prop_name == "pad-prefix") continue;
      elem_desc += ParseProperty(it->first, it->second) + " ";
    }
    elem_desc += "! " + pad_prefix + std::to_string(i) + "  ";
    elem_descs_.push_back(elem_desc);
  }
}

std::string YamlParser::ParseProperty(const YAML::Node& name, const YAML::Node& value) {
  std::string prop_name = name.as<std::string>();
  std::string prop_value = value.as<std::string>();

  if (prop_name == "_link_to") {
    return "! " + prop_value + " ";
  }

  return prop_name + "=" + prop_value;
}
}  // namespace streamer
}  // namespace fastdeploy
