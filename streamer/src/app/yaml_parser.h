
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

  bool AddNvUriSrcBins(const YAML::Node& properties);
  void SetProperty(GstElement* elem, const YAML::Node& name,
                   const YAML::Node& value);
  bool AddElement(const std::string& name, const YAML::Node& properties);
  void LinkElements();
  bool LinkSourePads(GstElement* streammux);

  AppConfig app_config_;
  std::string config_file_;
  YAML::Node yaml_config_;
  GstElement* pipeline_;
  std::vector<GstElement*> source_bins_;
  std::vector<GstElement*> unlinked_elements_;

  std::vector<std::string> elem_descs_;
};
}  // namespace streamer
}  // namespace fastdeploy
