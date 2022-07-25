#include "fastdeploy/vision/ppdet/ppyoloe.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

PPYOLOE::PPYOLOE(const std::string& model_file, const std::string& params_file,
                 const std::string& config_file,
                 const RuntimeOption& custom_option,
                 const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PPYOLOE::Initialize() {
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

bool PPYOLOE::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
              << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (cfg["arch"].as<std::string>() != "YOLO") {
    FDERROR << "Require the arch of model is YOLO, but arch defined in "
                 "config file is "
              << cfg["arch"].as<std::string>() << "." << std::endl;
    return false;
  }
  processors_.push_back(std::make_shared<BGR2RGB>());

  for (const auto& op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = op["is_scale"].as<bool>();
      processors_.push_back(std::make_shared<Normalize>(mean, std, is_scale));
    } else if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size(),
               "Require size of target_size be 2, but now it's " +
                   std::to_string(target_size.size()) + ".");
      FDASSERT(!keep_ratio,
               "Only support keep_ratio is false while deploy "
               "PaddleDetection model.");
      int width = target_size[1];
      int height = target_size[0];
      processors_.push_back(
          std::make_shared<Resize>(width, height, -1.0, -1.0, interp, false));
    } else if (op_name == "Permute") {
      processors_.push_back(std::make_shared<HWC2CHW>());
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
                << std::endl;
      return false;
    }
  }
  return true;
}

bool PPYOLOE::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
                << "." << std::endl;
      return false;
    }
  }

  outputs->resize(2);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));

  // reshape to [1, c, h, w]
  (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);

  (*outputs)[1].Allocate({1, 2}, FDDataType::FP32, InputInfoOfRuntime(1).name);
  float* ptr = static_cast<float*>((*outputs)[1].MutableData());
  ptr[0] = mat->Height() * 1.0 / mat->Height();
  ptr[1] = mat->Width() * 1.0 / mat->Width();
  return true;
}

bool PPYOLOE::Postprocess(std::vector<FDTensor>& infer_result,
                          DetectionResult* result, float conf_threshold,
                          float nms_threshold) {
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");
  int box_num = 0;
  if (infer_result[1].dtype == FDDataType::INT32) {
    box_num = *(static_cast<int32_t*>(infer_result[1].Data()));
  } else if (infer_result[1].dtype == FDDataType::INT64) {
    box_num = *(static_cast<int64_t*>(infer_result[1].Data()));
  } else {
    FDASSERT(
        false,
        "The output box_num of PPYOLOE model should be type of int32/int64.");
  }
  result->Reserve(box_num);
  float* box_data = static_cast<float*>(infer_result[0].Data());
  for (size_t i = 0; i < box_num; ++i) {
    if (box_data[i * 6 + 1] < conf_threshold) {
      continue;
    }
    result->label_ids.push_back(box_data[i * 6]);
    result->scores.push_back(box_data[i * 6 + 1]);
    result->boxes.emplace_back(
        std::array<float, 4>{box_data[i * 6 + 2], box_data[i * 6 + 3],
                             box_data[i * 6 + 4] - box_data[i * 6 + 2],
                             box_data[i * 6 + 5] - box_data[i * 6 + 3]});
  }
  return true;
}

bool PPYOLOE::Predict(cv::Mat* im, DetectionResult* result,
                      float conf_threshold, float iou_threshold) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data;
  if (!Preprocess(&mat, &processed_data)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  std::vector<FDTensor> infer_result;
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  if (!Postprocess(infer_result, result, conf_threshold, iou_threshold)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
