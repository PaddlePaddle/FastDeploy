#include "app/yaml_parser.h"
#include "gstreamer/source_bin.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace streamer {

YamlParser::YamlParser(const std::string& config_file) {
  config_file_ = config_file;
  // FDASSERT(BuildPipelineFromConfig(config_file),
          //  "Failed to create PaddleClasPreprocessor.");
}

void YamlParser::ValidateConfig(const std::string& config_file) {

}

void YamlParser::ParseAppConfg(AppConfig& app_config) {
  ValidateConfig(config_file_);
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
  }
  auto elem = cfg["app"];
  app_config.type = AppType::VIDEO_ANALYTICS;
  app_config.enable_perf_measurement = elem["enable-perf-measurement"].as<bool>();
  if (app_config.enable_perf_measurement) {
    app_config.perf_interval_sec = elem["perf-measurement-interval-sec"].as<int>();
  }
  app_config_ = app_config;
}

bool YamlParser::AddNvUriSrcBins(const YAML::Node& properties) {
  auto uri_list = properties["uri-list"].as<std::vector<std::string>>();
  auto gpu_id = properties["gpu-id"].as<int>();
  for (size_t i = 0; i < uri_list.size(); i++) {
    FDINFO << "Adding source " << uri_list[i] << std::endl;
    GstElement* source_bin = create_source_bin(i, const_cast<char*>(uri_list[i].c_str()), gpu_id);
    if (!source_bin) {
      g_printerr("Failed to create source bin. Exiting.\n");
      return false;
    }
    gst_bin_add(GST_BIN(pipeline_), source_bin);
    source_bins.push_back(source_bin);
  }
  return true;
}

bool YamlParser::BuildPipelineFromConfig(GstElement* pipeline) {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }
  pipeline_ = pipeline;
  for (const auto& elem : cfg) {
    std::string elem_name = elem.first.as<std::string>();
    std::cout << elem_name << std::endl;

    if (elem_name == "app") {
      // Do nothing
    } else if (elem_name == "nvurisrcbin_list") {
      AddNvUriSrcBins(elem.second);
    // } else if (elem_name == "nvstreammux") {
    //   NvStreamMuxConfig config;
    //   config.gpu_id = elem.second["gpu-id"].as<int>();
    //   config.batch_size = elem.second["batch-size"].as<int>();
    //   config.width = elem.second["width"].as<int>();
    //   config.height = elem.second["height"].as<int>();
    //   config.batched_push_timeout = elem.second["batched-push-timeout"].as<int>();
    //   AddNvStreamMux(pipeline, config);
    // } else if (elem_name == "nvinfer") {
    //   NvInferConfig config;
    //   config.gpu_id = elem.second["gpu-id"].as<int>();
    //   config.config_file_path = elem.second["config-file-path"].as<std::string>();
    //   AddNvInfer(pipeline, config);
    // } else if (elem_name == "nvtracker") {
    //   NvTrackerConfig config;
    //   config.gpu_id = elem.second["gpu-id"].as<int>();
    //   config.tracker_width = elem.second["tracker-width"].as<int>();
    //   config.tracker_height = elem.second["tracker-height"].as<int>();
    //   config.ll_lib_file = elem.second["ll-lib-file"].as<std::string>();
    //   config.ll_config_file = elem.second["ll-config-file"].as<std::string>();
    //   config.enable_batch_process = elem.second["enable-batch-process"].as<bool>();
    //   AddNvTracker(pipeline, config);
    // } else if (elem_name == "nvmultistreamtiler") {
    //   NvMultiStreamTilerConfig config;
    //   config.gpu_id = elem.second["gpu-id"].as<int>();
    //   config.rows = elem.second["rows"].as<int>();
    //   config.columns = elem.second["columns"].as<int>();
    //   AddNvMultiStreamTiler(pipeline, config);
    // } else if (elem_name == "nvosdbin") {
    //   NvOsdBinConfig config;
    //   config.gpu_id = elem.second["gpu-id"].as<int>();
    //   AddNvOsdBin(pipeline, config);
    // } else if (elem_name == "nvvideoencfilesinkbin") {
    //   NvVideoEncFileSinkBin config;
    //   config.gpu_id = elem.second["gpu-id"].as<int>();
    //   config.bitrate = elem.second["bitrate"].as<int>();
    //   config.output_file = elem.second["output-file"].as<std::string>();
    //   AddNvVideoEncFileSinkBin(pipeline, config);
    } else {
      FDASSERT(false, "Unsupported element: %s.", elem_name.c_str());
    }
  }
  // LinkElements();
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
