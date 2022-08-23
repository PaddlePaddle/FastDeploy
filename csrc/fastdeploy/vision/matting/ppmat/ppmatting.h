#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace matting {

class FASTDEPLOY_DECL PPMatting : public FastDeployModel {
 public:
  PPMatting(const std::string& model_file, const std::string& params_file,
            const std::string& config_file,
            const RuntimeOption& custom_option = RuntimeOption(),
            const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "PaddleMat"; }

  virtual bool Predict(cv::Mat* im, MattingResult* result);

  bool with_softmax = false;

  bool is_vertical_screen = false;

 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<int, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result, MattingResult* result,
                   std::map<std::string, std::array<int, 2>>* im_info);

  bool is_resized = false;

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
};

void FDTensor2FP32CVMat(cv::Mat& mat, FDTensor& infer_result,
                        bool contain_score_map);
}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy
