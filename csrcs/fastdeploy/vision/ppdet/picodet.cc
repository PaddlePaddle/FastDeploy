#include "fastdeploy/vision/ppdet/picodet.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

PicoDet::PicoDet(const std::string& model_file, const std::string& params_file,
                 const std::string& config_file,
                 const RuntimeOption& custom_option,
                 const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  background_label = -1;
  keep_top_k = 100;
  nms_eta = 1;
  nms_threshold = 0.6;
  nms_top_k = 1000;
  normalized = true;
  score_threshold = 0.025;
  CheckIfContainDecodeAndNMS();
  initialized = Initialize();
}

bool PicoDet::CheckIfContainDecodeAndNMS() {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (cfg["arch"].as<std::string>() == "PicoDet") {
    FDERROR << "The arch in config file is PicoDet, which means this model "
               "doesn contain box decode and nms, please export model with "
               "decode and nms."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
