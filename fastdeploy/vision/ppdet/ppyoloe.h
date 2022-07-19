#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

class FASTDEPLOY_DECL PPYOLOE : public FastDeployModel {
 public:
  PPYOLOE(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "PaddleDetection/PPYOLOE"; }

  virtual bool Initialize();

  virtual bool BuildPreprocessPipelineFromConfig();

  virtual bool Preprocess(Mat* mat, std::vector<FDTensor>* outputs);

  virtual bool Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result, float conf_threshold,
                           float nms_threshold);

  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.5, float nms_threshold = 0.7);

 private:
  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  // PaddleDetection can export model without nms
  // This flag will help us to handle the different
  // situation
  bool has_nms_;
};
}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
