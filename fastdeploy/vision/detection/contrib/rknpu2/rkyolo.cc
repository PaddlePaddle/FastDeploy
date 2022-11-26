#include "fastdeploy/vision/detection/contrib/rknpu2/rkyolo.h"

namespace fastdeploy {
namespace vision {
namespace detection {

RKYOLO::RKYOLO(const std::string& model_file,
               const fastdeploy::RuntimeOption& custom_option,
               const fastdeploy::ModelFormat& model_format) {
  if (model_format == ModelFormat::RKNN) {
    valid_cpu_backends = {};
    valid_gpu_backends = {};
    valid_rknpu_backends = {Backend::RKNPU2};
  } else {
    FDERROR << "RKYOLO Only Support run in RKNPU2" << std::endl;
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  initialized = Initialize();
}

bool RKYOLO::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  auto size = GetPreprocessor().GetSize();
  GetPostprocessor().SetHeightAndWeight(size[0],size[1]);
  return true;
}

bool RKYOLO::Predict(cv::Mat* im, DetectionResult* result, float conf_threshold, float nms_threshold) {
//  postprocessor_.SetConfThreshold(conf_threshold);
//  postprocessor_.SetNMSThreshold(nms_threshold);
  if (!Predict(*im, result)) {
    return false;
  }
  return true;
}

bool RKYOLO::Predict(const cv::Mat& im,
                     DetectionResult* result) {
  std::vector<DetectionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool RKYOLO::BatchPredict(const std::vector<cv::Mat>& images,
                          std::vector<DetectionResult>* results) {
  std::vector<FDMat> fd_images = WrapMat(images);

  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }
  auto pad_hw_values_ = preprocessor_.GetPadHWValues();
  postprocessor_.SetPadHWValues(preprocessor_.GetPadHWValues());
  std::cout << "preprocessor_ scale_ = " << preprocessor_.GetScale()[0] << std::endl;
  postprocessor_.SetScale(preprocessor_.GetScale());

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
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

} // namespace detection
} // namespace vision
} // namespace fastdeploy