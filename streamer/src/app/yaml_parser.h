
#pragma once

#include "yaml-cpp/yaml.h"

#include <gst/gst.h>

namespace fastdeploy {
namespace streamer {

enum AppType {
  VIDEO_ANALYTICS,  ///< Video analytics app
};

struct AppConfig {
  AppType type;
  bool enable_perf_measurement = false;
  int perf_interval_sec = 5;
};

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

  bool BuildPipelineFromConfig(GstElement* pipeline);

  GstElement* pipeline;
  std::vector<GstElement*> source_bins;

 private:
  bool AddNvUriSrcBins(const YAML::Node& properties);
  void SetProperty(GstElement* elem, const YAML::Node& name,
                   const YAML::Node& value);
  bool AddElement(const std::string& name, const YAML::Node& properties);

  AppConfig app_config_;
  std::string config_file_;
  GstElement* pipeline_;
  YAML::Node yaml_config_;
};
}  // namespace streamer
}  // namespace fastdeploy
