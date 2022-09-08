#include "fastdeploy/vision/detection/ppdet/ppyoloe.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

namespace fastdeploy {
namespace vision {
namespace detection {

PPYOLOE::PPYOLOE(const std::string& model_file, const std::string& params_file,
                 const std::string& config_file,
                 const RuntimeOption& custom_option,
                 const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

void PPYOLOE::GetNmsInfo() {
  if (runtime_option.model_format == Frontend::PADDLE) {
    std::string contents;
    if (!ReadBinaryFromFile(runtime_option.model_file, &contents)) {
      return;
    }
    auto reader = paddle2onnx::PaddleReader(contents.c_str(), contents.size());
    if (reader.has_nms) {
      has_nms_ = true;
      background_label = reader.nms_params.background_label;
      keep_top_k = reader.nms_params.keep_top_k;
      nms_eta = reader.nms_params.nms_eta;
      nms_threshold = reader.nms_params.nms_threshold;
      score_threshold = reader.nms_params.score_threshold;
      nms_top_k = reader.nms_params.nms_top_k;
      normalized = reader.nms_params.normalized;
    }
  }
}

bool PPYOLOE::Initialize() {
#ifdef ENABLE_PADDLE_FRONTEND
  // remove multiclass_nms3 now
  // this is a trick operation for ppyoloe while inference on trt
  GetNmsInfo();
  runtime_option.remove_multiclass_nms_ = true;
  runtime_option.custom_op_info_["multiclass_nms3"] = "MultiClassNMS";
#endif
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  if (has_nms_ && runtime_option.backend == Backend::TRT) {
    FDINFO << "Detected operator multiclass_nms3 in your model, will replace "
              "it with fastdeploy::backend::MultiClassNMS(background_label="
           << background_label << ", keep_top_k=" << keep_top_k
           << ", nms_eta=" << nms_eta << ", nms_threshold=" << nms_threshold
           << ", score_threshold=" << score_threshold
           << ", nms_top_k=" << nms_top_k << ", normalized=" << normalized
           << ")." << std::endl;
    has_nms_ = false;
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

  processors_.push_back(std::make_shared<BGR2RGB>());

  bool has_permute = false;
  for (const auto& op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = true;
      if (op["is_scale"]) {
        is_scale = op["is_scale"].as<bool>();
      }
      std::string norm_type = "mean_std";
      if (op["norm_type"]) {
        norm_type = op["norm_type"].as<std::string>();
      }
      if (norm_type != "mean_std") {
        std::fill(mean.begin(), mean.end(), 0.0);
        std::fill(std.begin(), std.end(), 1.0);
      }
      processors_.push_back(std::make_shared<Normalize>(mean, std, is_scale));
    } else if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size(),
               "Require size of target_size be 2, but now it's %lu.",
               target_size.size());
      if (!keep_ratio) {
        int width = target_size[1];
        int height = target_size[0];
        processors_.push_back(
            std::make_shared<Resize>(width, height, -1.0, -1.0, interp, false));
      } else {
        int min_target_size = std::min(target_size[0], target_size[1]);
        int max_target_size = std::max(target_size[0], target_size[1]);
        processors_.push_back(std::make_shared<ResizeByShort>(
            min_target_size, interp, true, max_target_size));
      }
    } else if (op_name == "Permute") {
      // Do nothing, do permute as the last operation
      has_permute = true;
      continue;
      // processors_.push_back(std::make_shared<HWC2CHW>());
    } else if (op_name == "Pad") {
      auto size = op["size"].as<std::vector<int>>();
      auto value = op["fill_value"].as<std::vector<float>>();
      processors_.push_back(std::make_shared<Cast>("float"));
      processors_.push_back(
          std::make_shared<PadToSize>(size[1], size[0], value));
    } else if (op_name == "PadStride") {
      auto stride = op["stride"].as<int>();
      processors_.push_back(
          std::make_shared<StridePad>(stride, std::vector<float>(3, 0)));
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  if (has_permute) {
    processors_.push_back(std::make_shared<Permute>());
  } else {
    processors_.push_back(std::make_shared<HWC2CHW>());
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
  ptr[0] = mat->Height() * 1.0 / origin_h;
  ptr[1] = mat->Width() * 1.0 / origin_w;
  return true;
}

bool PPYOLOE::Postprocess(std::vector<FDTensor>& infer_result,
                          DetectionResult* result) {
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");

  if (!has_nms_) {
    int boxes_index = 0;
    int scores_index = 1;
    if (infer_result[0].shape[1] == infer_result[1].shape[2]) {
      boxes_index = 0;
      scores_index = 1;
    } else if (infer_result[0].shape[2] == infer_result[1].shape[1]) {
      boxes_index = 1;
      scores_index = 0;
    } else {
      FDERROR << "The shape of boxes and scores should be [batch, boxes_num, "
                 "4], [batch, classes_num, boxes_num]"
              << std::endl;
      return false;
    }

    backend::MultiClassNMS nms;
    nms.background_label = background_label;
    nms.keep_top_k = keep_top_k;
    nms.nms_eta = nms_eta;
    nms.nms_threshold = nms_threshold;
    nms.score_threshold = score_threshold;
    nms.nms_top_k = nms_top_k;
    nms.normalized = normalized;
    nms.Compute(static_cast<float*>(infer_result[boxes_index].Data()),
                static_cast<float*>(infer_result[scores_index].Data()),
                infer_result[boxes_index].shape,
                infer_result[scores_index].shape);
    if (nms.out_num_rois_data[0] > 0) {
      result->Reserve(nms.out_num_rois_data[0]);
    }
    for (size_t i = 0; i < nms.out_num_rois_data[0]; ++i) {
      result->label_ids.push_back(nms.out_box_data[i * 6]);
      result->scores.push_back(nms.out_box_data[i * 6 + 1]);
      result->boxes.emplace_back(std::array<float, 4>{
          nms.out_box_data[i * 6 + 2], nms.out_box_data[i * 6 + 3],
          nms.out_box_data[i * 6 + 4], nms.out_box_data[i * 6 + 5]});
    }
  } else {
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
      result->label_ids.push_back(box_data[i * 6]);
      result->scores.push_back(box_data[i * 6 + 1]);
      result->boxes.emplace_back(
          std::array<float, 4>{box_data[i * 6 + 2], box_data[i * 6 + 3],
                               box_data[i * 6 + 4], box_data[i * 6 + 5]});
    }
  }
  return true;
}

bool PPYOLOE::Predict(cv::Mat* im, DetectionResult* result) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data;
  if (!Preprocess(&mat, &processed_data)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  float* tmp = static_cast<float*>(processed_data[1].Data());
  std::vector<FDTensor> infer_result;
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  if (!Postprocess(infer_result, result)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
