#include "fastdeploy/vision/detection/ppdet/pptinypose.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

namespace fastdeploy {
namespace vision {
namespace detection {

PPTinyPose::PPTinyPose(const std::string& model_file,
                       const std::string& params_file,
                       const std::string& config_file,
                       const RuntimeOption& custom_option,
                       const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PPTinyPose::Initialize() {
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

bool PPTinyPose::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  std::string arch = cfg["arch"].as<std::string>();
  if (arch != "HRNet" && arch != "HigherHRNet") {
    FDERROR << "Require the arch of model is HRNet or HigherHRNet, but arch "
               "defined in "
               "config file is "
            << arch << "." << std::endl;
    return false;
  }

  // Get draw_threshold for visualization
  if (cfg["draw_threshold"].IsDefined()) {
    threshold = cfg["draw_threshold"].as<float>();
  }
  processors_.push_back(std::make_shared<BGR2RGB>());

  for (const auto& op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = op["is_scale"].as<bool>();
      processors_.push_back(std::make_shared<Normalize>(mean, std, is_scale));
    } else if (op_name == "Permute") {
      processors_.push_back(std::make_shared<HWC2CHW>());
    } else if (op_name == "TopDownEvalAffine") {
      auto trainsize = op["trainsize"].as<std::vector<int>>();
      int height = trainsize[1];
      int width = trainsize[0];
      processors_.push_back(
          std::make_shared<Resize>(width, height, -1.0, -1.0, 1, false));
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  return true;
}

bool PPTinyPose::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  outputs->resize(1);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));

  // reshape to [1, c, h, w]
  (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);

  return true;
}

bool PPTinyPose::Postprocess(std::vector<FDTensor>& infer_result,
                             KeyPointDetectionResult* result,
                             const std::vector<float>& center,
                             const std::vector<float>& scale) {
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");
  result->Clear();

  // Calculate output length
  int outdata_size =
      std::accumulate(infer_result[0].shape.begin(),
                      infer_result[0].shape.end(), 1, std::multiplies<int>());
  int idxdata_size =
      std::accumulate(infer_result[1].shape.begin(),
                      infer_result[1].shape.end(), 1, std::multiplies<int>());
  if (outdata_size < 6) {
    FDWARNING << "PPTinyPose No object detected." << std::endl;
  }
  float* out_data = static_cast<float*>(infer_result[0].Data());
  void* idx_data = infer_result[1].Data();
  int idx_dtype = infer_result[1].dtype;
  std::vector<int> out_data_shape(infer_result[0].shape.begin(),
                                  infer_result[0].shape.end());
  std::vector<int> idx_data_shape(infer_result[1].shape.begin(),
                                  infer_result[1].shape.end());
  std::vector<float> preds(out_data_shape[1] * 3, 0);
  std::vector<float> heatmap(out_data, out_data + outdata_size);
  std::vector<int64_t> idxout(idxdata_size);
  if (idx_dtype == FDDataType::INT32) {
    std::copy(static_cast<int32_t*>(idx_data),
              static_cast<int32_t*>(idx_data) + idxdata_size, idxout.begin());
  } else if (idx_dtype == FDDataType::INT64) {
    std::copy(static_cast<int64_t*>(idx_data),
              static_cast<int64_t*>(idx_data) + idxdata_size, idxout.begin());
  } else {
    FDERROR << "Don't support inference output FDDataType." << std::endl;
  }
  GetFinalPredictions(heatmap, out_data_shape, idxout, center, scale, &preds,
                      this->use_dark);
  result->Reserve(outdata_size);
  result->num_joints = out_data_shape[1];
  result->keypoints.clear();
  for (int i = 0; i < out_data_shape[1]; i++) {
    result->keypoints.emplace_back(preds[i * 3 + 1]);
    result->keypoints.emplace_back(preds[i * 3 + 2]);
    result->keypoints.emplace_back(preds[i * 3]);
  }
  return true;
}

bool PPTinyPose::Predict(cv::Mat* im, KeyPointDetectionResult* result,
                         DetectionResult* detection_result) {
  std::vector<cv::Mat> crop_imgs;
  std::vector<std::vector<float>> center_bs;
  std::vector<std::vector<float>> scale_bs;
  int crop_imgs_num = 0;
  if (detection_result != nullptr) {
    int box_num = detection_result->boxes.size();
    for (int i = 0; i < box_num; i++) {
      auto box = detection_result->boxes[i];
      auto label_id = detection_result->label_ids[i];
      cv::Mat crop_img;
      std::vector<int> rect = {
          static_cast<int>(box[0]), static_cast<int>(box[1]),
          static_cast<int>(box[2]), static_cast<int>(box[3])};
      std::vector<float> center;
      std::vector<float> scale;
      if (label_id == 0) {
        utils::CropImage(*im, &crop_img, rect, &center, &scale);
        center_bs.emplace_back(center);
        scale_bs.emplace_back(scale);
        crop_imgs.emplace_back(crop_img);
        crop_imgs_num += 1;
      }
    }
  } else {
    cv::Mat crop_img;
    std::vector<int> rect = {0, 0, im->cols - 1, im->rows - 1};
    std::vector<float> center;
    std::vector<float> scale;
    utils::CropImage(*im, &crop_img, rect, &center, &scale);
    center_bs.emplace_back(center);
    scale_bs.emplace_back(scale);
    crop_imgs.emplace_back(crop_img);
    crop_imgs_num += 1;
  }
  for (int i = 0; i < crop_imgs_num; i++) {
    Mat mat(crop_imgs[i]);
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
    KeyPointDetectionResult one_cropimg_result;
    if (!Postprocess(infer_result, &one_cropimg_result, center_bs[i],
                     scale_bs[i])) {
      FDERROR << "Failed to postprocess while using model:" << ModelName()
              << "." << std::endl;
      return false;
    }
    if (result->num_joints == -1) {
      result->num_joints = one_cropimg_result.num_joints;
    }
    std::copy(one_cropimg_result.keypoints.begin(),
              one_cropimg_result.keypoints.end(),
              std::back_inserter(result->keypoints));
  }

  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
