#include "fastdeploy/vision/detection/ppdet/base.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace detection {

PPDetBase::PPDetBase(const std::string& model_file, const std::string& params_file,
             const std::string& config_file,
             const RuntimeOption& custom_option,
             const ModelFormat& model_format) : preprocessor_(config_file) {
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
}

bool PPDetBase::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool PPDetBase::Predict(cv::Mat* im, DetectionResult* result) {
  return Predict(*im, result);
}

bool PPDetBase::Predict(const cv::Mat& im, DetectionResult* result) {
  std::vector<DetectionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool PPDetBase::BatchPredict(const std::vector<cv::Mat>& imgs, std::vector<DetectionResult>* results) {
  std::vector<FDMat> fd_images = WrapMat(imgs);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }
  reused_input_tensors_[0].name = "image";
  reused_input_tensors_[1].name = "scale_factor";
  reused_input_tensors_[2].name = "im_shape";
  // Some models don't need im_shape as input
  if (NumInputsOfRuntime() == 2) {
    reused_input_tensors_.pop_back();
  }

  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime." << std::endl;
    return false;
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
