#include "fastdeploy/vision/matting/ppmatting/ppmatting.h"
#include "fastdeploy/vision.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace matting {

PPMatting::PPMatting(const std::string& model_file,
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

bool PPMatting::Initialize() {
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

bool PPMatting::BuildPreprocessPipelineFromConfig() {
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
      if (op["type"].as<std::string>() == "LimitShort") {
        int max_short = -1;
        int min_short = -1;
        if (op["max_short"]) {
          max_short = op["max_short"].as<int>();
        }
        if (op["min_short"]) {
          min_short = op["min_short"].as<int>();
        }
        processors_.push_back(
            std::make_shared<LimitShort>(max_short, min_short));
      } else if (op["type"].as<std::string>() == "ResizeToIntMult") {
        int mult_int = 32;
        if (op["mult_int"]) {
          mult_int = op["mult_int"].as<int>();
        }
        processors_.push_back(std::make_shared<ResizeToIntMult>(mult_int));
      } else if (op["type"].as<std::string>() == "Normalize") {
        std::vector<float> mean = {0.5, 0.5, 0.5};
        std::vector<float> std = {0.5, 0.5, 0.5};
        if (op["mean"]) {
          mean = op["mean"].as<std::vector<float>>();
        }
        if (op["std"]) {
          std = op["std"].as<std::vector<float>>();
        }
        processors_.push_back(std::make_shared<Normalize>(mean, std));
      }
    }
    processors_.push_back(std::make_shared<HWC2CHW>());
  }
  return true;
}

bool PPMatting::Preprocess(Mat* mat, FDTensor* output,
                           std::map<std::string, std::array<int, 2>>* im_info) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<int>(mat->Height()),
                                static_cast<int>(mat->Width())};

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  output->name = InputInfoOfRuntime(0).name;
  return true;
}

bool PPMatting::Postprocess(
    std::vector<FDTensor>& infer_result, MattingResult* result,
    const std::map<std::string, std::array<int, 2>>& im_info) {
  FDASSERT((infer_result.size() == 1),
           "The default number of output tensor must be 1 according to "
           "modnet.");
  FDTensor& alpha_tensor = infer_result.at(0);  // (1,h,w,1)
  FDASSERT((alpha_tensor.shape[0] == 1), "Only support batch =1 now.");
  if (alpha_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  // 先获取alpha并resize (使用opencv)
  auto iter_ipt = im_info.find("input_shape");
  auto iter_out = im_info.find("output_shape");
  FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  int out_h = iter_out->second[0];
  int out_w = iter_out->second[1];
  int ipt_h = iter_ipt->second[0];
  int ipt_w = iter_ipt->second[1];

  // TODO: 需要修改成FDTensor或Mat的运算 现在依赖cv::Mat
  float* alpha_ptr = static_cast<float*>(alpha_tensor.Data());
  cv::Mat alpha_zero_copy_ref(out_h, out_w, CV_32FC1, alpha_ptr);
  Mat alpha_resized(alpha_zero_copy_ref);  // ref-only, zero copy.
  if ((out_h != ipt_h) || (out_w != ipt_w)) {
    // already allocated a new continuous memory after resize.
    // cv::resize(alpha_resized, alpha_resized, cv::Size(ipt_w, ipt_h));
    Resize::Run(&alpha_resized, ipt_w, ipt_h, -1, -1);
  }

  result->Clear();
  // note: must be setup shape before Resize
  result->contain_foreground = false;
  // 和输入原图大小对应的alpha
  result->shape = {static_cast<int64_t>(ipt_h), static_cast<int64_t>(ipt_w)};
  int numel = ipt_h * ipt_w;
  int nbytes = numel * sizeof(float);
  result->Resize(numel);
  std::memcpy(result->alpha.data(), alpha_resized.GetCpuMat()->data, nbytes);
  return true;
}

bool PPMatting::Predict(cv::Mat* im, MattingResult* result) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data(1);

  std::map<std::string, std::array<int, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<int>(mat.Height()),
                            static_cast<int>(mat.Width())};
  im_info["output_shape"] = {static_cast<int>(mat.Height()),
                             static_cast<int>(mat.Width())};

  if (!Preprocess(&mat, &(processed_data[0]), &im_info)) {
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
  if (!Postprocess(infer_result, result, im_info)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy
