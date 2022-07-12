
#include "fastdeploy/vision/ppcls/model.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace ppcls {

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
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }
  auto preprocess_cfg = cfg["PreProcess"]["transform_ops"];
  processors_.push_back(std::make_shared<BGR2RGB>());
  for (const auto& op : preprocess_cfg) {
    FDASSERT(op.IsMap(),
             "Require the transform information in yaml be Map type.");
    auto op_name = op.begin()->first.as<std::string>();
    if (op_name == "ResizeImage") {
      int target_size = op.begin()->second["resize_short"].as<int>();
      bool use_scale = false;
      int interp = 1;
      processors_.push_back(
          std::make_shared<ResizeByShort>(target_size, 1, use_scale));
    } else if (op_name == "CropImage") {
      int width = op.begin()->second["size"].as<int>();
      int height = op.begin()->second["size"].as<int>();
      processors_.push_back(std::make_shared<CenterCrop>(width, height));
    } else if (op_name == "NormalizeImage") {
      auto mean = op.begin()->second["mean"].as<std::vector<float>>();
      auto std = op.begin()->second["std"].as<std::vector<float>>();
      auto scale = op.begin()->second["scale"].as<float>();
      FDASSERT((scale - 0.00392157) < 1e-06 && (scale - 0.00392157) > -1e-06,
               "Only support scale in Normalize be 0.00392157, means the pixel "
               "is in range of [0, 255].");
      processors_.push_back(std::make_shared<Normalize>(mean, std));
    } else if (op_name == "ToCHWImage") {
      processors_.push_back(std::make_shared<HWC2CHW>());
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
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

bool Model::Postprocess(const FDTensor& infer_result, ClassifyResult* result,
                        int topk) {
  int num_classes = infer_result.shape[1];
  const float* infer_result_buffer =
      reinterpret_cast<const float*>(infer_result.data.data());
  topk = std::min(num_classes, topk);
  result->label_ids =
      utils::TopKIndices(infer_result_buffer, num_classes, topk);
  result->scores.resize(topk);
  for (int i = 0; i < topk; ++i) {
    result->scores[i] = *(infer_result_buffer + result->label_ids[i]);
  }
  return true;
}

bool Model::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
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

  if (!Postprocess(infer_result[0], result, topk)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace ppcls
}  // namespace vision
}  // namespace fastdeploy
