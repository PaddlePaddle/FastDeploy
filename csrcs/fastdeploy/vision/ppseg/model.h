#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace ppseg {

class FASTDEPLOY_DECL Model : public FastDeployModel {
 public:
  Model(const std::string& model_file, const std::string& params_file,
        const std::string& config_file,
        const RuntimeOption& custom_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "ppseg"; }

  virtual bool Predict(cv::Mat* im, SegmentationResult* result);

  bool with_softmax = false;

  bool is_vertical_screen = false;

 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<int, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result, SegmentationResult* result,
                   std::map<std::string, std::array<int, 2>>* im_info);

  bool is_resized = false;

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
};
}  // namespace ppseg
}  // namespace vision
}  // namespace fastdeploy
