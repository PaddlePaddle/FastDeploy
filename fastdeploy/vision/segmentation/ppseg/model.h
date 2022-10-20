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
                 const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "PaddleSeg"; }

  virtual bool Predict(cv::Mat* im, SegmentationResult* result);

  bool apply_softmax = false;

  bool is_vertical_screen = false;

  // RKNPU2 can run normalize and hwc2chw on the NPU.
  // This function is used to close normalize and hwc2chw operations in preprocessing.
  void DisableNormalizeAndPermute();
 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs);

  bool Postprocess(FDTensor* infer_result, SegmentationResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);

  bool is_with_softmax = false;

  bool is_with_argmax = true;

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  
  // for recording the switch of normalize and hwc2chw
  bool switch_of_nor_and_per = true;  
};

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
