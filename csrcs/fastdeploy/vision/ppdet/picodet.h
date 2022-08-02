#pragma once
#include "fastdeploy/vision/ppdet/ppyoloe.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

class FASTDEPLOY_DECL PicoDet : public PPYOLOE {
 public:
  PicoDet(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const Frontend& model_format = Frontend::PADDLE);

  // Only support picodet contains decode and nms
  bool CheckIfContainDecodeAndNMS();

  virtual std::string ModelName() const { return "PaddleDetection/PicoDet"; }
};
}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
