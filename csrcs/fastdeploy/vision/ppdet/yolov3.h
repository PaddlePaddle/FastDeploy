#pragma once
#include "fastdeploy/vision/ppdet/ppyoloe.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

class FASTDEPLOY_DECL YOLOv3 : public PPYOLOE {
 public:
  YOLOv3(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const Frontend& model_format = Frontend::PADDLE);

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv3"; }

  virtual bool Preprocess(Mat* mat, std::vector<FDTensor>* outputs);
};
}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
