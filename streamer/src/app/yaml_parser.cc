#include "app/yaml_parser.h"
#include "gstreamer/source_bin.h"
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
  } else if (type_name == "GstCaps") {
    auto caps_str = value.as<std::string>();
    GstCaps* caps = gst_caps_from_string(caps_str.c_str());
    g_object_set(G_OBJECT(elem), prop_name.c_str(), caps, NULL);
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
  } else {
    unlinked_elements_.insert(unlinked_elements_.begin(), source_bins_[0]);
  }
  for (size_t i = 1; i < unlinked_elements_.size(); i++) {
    FDASSERT(
        gst_element_link(unlinked_elements_[i - 1], unlinked_elements_[i]),
        "Failed to link elements %s and %s.",
        GetElementName(unlinked_elements_[i - 1]).c_str(),
        GetElementName(unlinked_elements_[i]).c_str());
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
