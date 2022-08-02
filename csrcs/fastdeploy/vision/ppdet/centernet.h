#pragma once
#include "fastdeploy/vision/ppdet/ppyolo.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

class FASTDEPLOY_DECL CenterNet : public PPYOLO {
 public:
  CenterNet(const std::string& model_file, const std::string& params_file,
            const std::string& config_file,
            const RuntimeOption& custom_option = RuntimeOption(),
            const Frontend& model_format = Frontend::PADDLE);

  virtual std::string ModelName() const { return "PaddleDetection/CenterNet"; }
};
}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
