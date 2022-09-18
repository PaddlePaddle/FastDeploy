#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace segmentation {

class FASTDEPLOY_DECL PaddleSegModel : public FastDeployModel {
 public:
  PaddleSegModel(const std::string& model_file, const std::string& params_file,
                 const std::string& config_file,
                 const RuntimeOption& custom_option = RuntimeOption(),
                 const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "PaddleSeg"; }

  virtual bool Predict(cv::Mat* im, SegmentationResult* result);

  bool apply_softmax = false;

  bool is_vertical_screen = false;

 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs);

  bool Postprocess(FDTensor* infer_result, SegmentationResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);

  bool is_resized = false;

  bool is_with_softmax = false;

  bool is_with_argmax = true;

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
};

void FDTensor2FP32CVMat(cv::Mat* mat, FDTensor& infer_result,
                        const bool contain_score_map);
}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
