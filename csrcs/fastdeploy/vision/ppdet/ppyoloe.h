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
                           DetectionResult* result);

  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.5, float nms_threshold = 0.7);

 private:
  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  // configuration for nms
  int64_t background_label = -1;
  int64_t keep_top_k = 300;
  float nms_eta = 1.0;
  float nms_threshold = 0.7;
  float score_threshold = 0.01;
  int64_t nms_top_k = 10000;
  bool normalized = true;
  bool has_nms_ = false;

  // This function will used to check if this model contains multiclass_nms
  // and get parameters from the operator
  void GetNmsInfo();
};
}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
