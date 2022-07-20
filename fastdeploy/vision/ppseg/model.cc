#include "fastdeploy/vision/ppseg/model.h"
#include "fastdeploy/vision.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace ppseg {

Model::Model(const std::string& model_file, const std::string& params_file,
             const std::string& config_file, const RuntimeOption& custom_option,
             const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool Model::Initialize() {
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool Model::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  processors_.push_back(std::make_shared<BGR2RGB>());
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (cfg["Deploy"]["transforms"]) {
    auto preprocess_cfg = cfg["Deploy"]["transforms"];
    for (const auto& op : preprocess_cfg) {
      FDASSERT(op.IsMap(),
               "Require the transform information in yaml be Map type.");
      if (op["type"].as<std::string>() == "Normalize") {
        std::vector<float> mean = {0.5, 0.5, 0.5};
        std::vector<float> std = {0.5, 0.5, 0.5};
        if (op["mean"]) {
          mean = op["mean"].as<std::vector<float>>();
        }
        if (op["std"]) {
          std = op["std"].as<std::vector<float>>();
        }
        processors_.push_back(std::make_shared<Normalize>(mean, std));

      } else if (op["type"].as<std::string>() == "Resize") {
        const auto& target_size = op["target_size"];
        int resize_width = target_size[0].as<int>();
        int resize_height = target_size[1].as<int>();
        processors_.push_back(
            std::make_shared<Resize>(resize_width, resize_height));
      }
    }
    processors_.push_back(std::make_shared<HWC2CHW>());
  }
  return true;
}

bool Model::Preprocess(Mat* mat, FDTensor* output) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }
  int channel = mat->Channels();
  int width = mat->Width();
  int height = mat->Height();
  output->name = InputInfoOfRuntime(0).name;
  output->SetExternalData({1, channel, height, width}, FDDataType::FP32,
                          mat->GetCpuMat()->ptr());
  return true;
}

bool Model::Postprocess(const FDTensor& infer_result,
                        SegmentationResult* result) {
  result->Clear();
  std::vector<int64_t> output_shape = infer_result.shape;
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  const int64_t* infer_result_buffer =
      reinterpret_cast<const int64_t*>(infer_result.data.data());
  int64_t height = output_shape[1];
  int64_t width = output_shape[2];
  result->Resize(height, width);
  for (int64_t i = 0; i < height; i++) {
    int64_t begin = i * width;
    int64_t end = (i + 1) * width - 1;
    std::copy(infer_result_buffer + begin, infer_result_buffer + end,
              result->masks[i].begin());
  }

  return true;
}

bool Model::Predict(cv::Mat* im, SegmentationResult* result) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data(1);
  if (!Preprocess(&mat, &(processed_data[0]))) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  std::vector<FDTensor> infer_result(1);
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  if (!Postprocess(infer_result[0], result)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace ppseg
}  // namespace vision
}  // namespace fastdeploy
