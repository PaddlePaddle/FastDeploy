#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/common/processors/transform.h"

namespace fastdeploy {
namespace vision {
namespace ppcls {

class FASTDEPLOY_DECL Model : public FastDeployModel {
 public:
  Model(const std::string& model_file, const std::string& params_file,
        const std::string& config_file,
        const RuntimeOption& custom_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "ppclas-classify"; }

  // TODO(jiangjiajun) Batch is on the way
  virtual bool Predict(cv::Mat* im, ClassifyResult* result, int topk = 1);

 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs);

  bool Postprocess(const FDTensor& infer_result, ClassifyResult* result,
                   int topk = 1);

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
};
} // namespace ppcls
} // namespace vision
} // namespace fastdeploy
