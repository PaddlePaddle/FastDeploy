#include "fastdeploy/vision/ppdet/centernet.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

CenterNet::CenterNet(const std::string& model_file,
                     const std::string& params_file,
                     const std::string& config_file,
                     const RuntimeOption& custom_option,
                     const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};
  has_nms_ = true;
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
