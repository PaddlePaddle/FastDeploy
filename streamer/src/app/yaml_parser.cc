#include "app/yaml_parser.h"
#include "gstreamer/source_bin.h"
#include "gstreamer/utils.h"

namespace fastdeploy {
namespace streamer {

YamlParser::YamlParser(const std::string& config_file) {
  // FDASSERT(BuildPipelineFromConfig(config_file),
          //  "Failed to create PaddleClasPreprocessor.");
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
  app_config.type = AppType::VIDEO_ANALYTICS;
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

bool YamlParser::BuildPipelineFromConfig(GstElement* pipeline) {
  pipeline_ = pipeline;
  for (const auto& elem : yaml_config_) {
    std::string elem_name = elem.first.as<std::string>();
    std::cout << elem_name << std::endl;
    FDASSERT(AddElement(elem_name, elem.second), "Failed to add element: %s",
             elem_name.c_str());
  }
  LinkElements();
  return true;
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
    source_bins_.push_back(source_bin);
  }
  return true;
}

void YamlParser::SetProperty(GstElement* elem, const YAML::Node& name,
                             const YAML::Node& value) {
  std::string prop_name = name.as<std::string>();
  std::cout << "Setting property " << prop_name << std::endl;

  // Get default property value by property name,
  // then get the value's data type.
  GValue default_value = G_VALUE_INIT;
  g_object_get_property(G_OBJECT(elem), prop_name.c_str(), &default_value);
  std::string type_name(g_type_name(default_value.g_type));
  std::cout << "  Type: " << type_name << std::endl;

  // Convert the value in YAML into the data type of the property value
  if (type_name == "gboolean") {
    auto prop_value = value.as<bool>();
    g_object_set(G_OBJECT(elem), prop_name.c_str(), prop_value, NULL);
  } else if (type_name == "guint") {
    auto prop_value = value.as<guint>();
    g_object_set(G_OBJECT(elem), prop_name.c_str(), prop_value, NULL);
  } else if (type_name == "gint") {
    auto prop_value = value.as<gint>();
    g_object_set(G_OBJECT(elem), prop_name.c_str(), prop_value, NULL);
  } else if (type_name == "gchararray") {
    auto prop_value = value.as<std::string>();
    g_object_set(G_OBJECT(elem), prop_name.c_str(), prop_value.c_str(), NULL);
  } else {
    FDASSERT(false, "Unsupported property value type: %s.", type_name.c_str());
  }
}

bool YamlParser::AddElement(const std::string& name, const YAML::Node& properties) {
  if (name == "app") return true;

  if (name == "nvurisrcbin_list") {
    return AddNvUriSrcBins(properties);
  }

  GstElement* elem = gst_element_factory_make(name.c_str(), NULL);
  for (auto it = properties.begin(); it != properties.end(); it++) {
    SetProperty(elem, it->first, it->second);
  }
  gst_bin_add(GST_BIN(pipeline_), elem);
  unlinked_elements_.push_back(elem);
  return true;
}

void YamlParser::LinkElements() {
  std::string elem_name = GetElementName(unlinked_elements_[0]);
  if (elem_name.rfind("nvstreammux", 0) == 0) {
    FDASSERT(LinkSourePads(unlinked_elements_[0]),
             "Failed to link source pads");
  }
  for (size_t i = 1; i < unlinked_elements_.size(); i++) {
    FDASSERT(
        gst_element_link(unlinked_elements_[i - 1], unlinked_elements_[i]),
        "Failed to link elements.");
  }
}

bool YamlParser::LinkSourePads(GstElement* streammux) {
  for (size_t i = 0; i < source_bins_.size(); i++) {
    std::string pad_name = "sink_" + std::to_string(i);
    GstPad* sinkpad = gst_element_get_request_pad(streammux, pad_name.c_str());
    if (!sinkpad) {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return false;
    }
    GstPad* srcpad = gst_element_get_static_pad(source_bins_[i], "src");
    if (!srcpad) {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return false;
    }
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return false;
    }
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
