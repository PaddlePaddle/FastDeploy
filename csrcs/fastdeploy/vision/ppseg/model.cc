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
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT};
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
        is_resized = true;
        processors_.push_back(
            std::make_shared<Resize>(resize_width, resize_height));
      }
    }
    processors_.push_back(std::make_shared<HWC2CHW>());
  }
  return true;
}

bool Model::Preprocess(Mat* mat, FDTensor* output,
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
  for (auto& i : output->shape) {
    std::cout << "Preprocess before shape: " << i << std::endl;
  }
  output->shape.insert(output->shape.begin(), 1);
  for (auto& i : output->shape) {
    std::cout << "Preprocess After shape: " << i << std::endl;
  }
  output->name = InputInfoOfRuntime(0).name;
  return true;
}

bool Model::Postprocess(FDTensor& infer_result, SegmentationResult* result,
                        std::map<std::string, std::array<int, 2>>* im_info) {
  FDASSERT(infer_result.dtype == FDDataType::INT64 ||
               infer_result.dtype == FDDataType::FP32,
           "Require the data type of output is int64 or fp32, but now it's " +
               Str(const_cast<fastdeploy::FDDataType&>(infer_result.dtype)) +
               ".");
  result->Clear();
  if (infer_result.shape.size() == 4) {
    FDASSERT(infer_result.shape[0] == 1, "Only support batch size = 1.");
    result->contain_score_map = true;
    float_t* infer_result_buffer =
        reinterpret_cast<float_t*>(infer_result.MutableData());
    // NCHW to NHWC
    int num = infer_result.shape[0];
    int channel = infer_result.shape[1];
    int height = infer_result.shape[2];
    int width = infer_result.shape[3];
    int chw = channel * height * width;
    int wc = width * channel;
    int wh = width * height;
    std::vector<float_t> hwc_data(chw);
    int index = 0;
    for (int n = 0; n < num; n++) {
      for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            hwc_data[n * chw + h * wc + w * channel + c] =
                *(infer_result_buffer + index);
            index++;
          }
        }
      }
    }
    std::memcpy(infer_result.MutableData(), hwc_data.data(),
                num * chw * sizeof(float_t));
    infer_result.shape = {num, height, width, channel};
  }

  FDTensor new_infer_result;
  Mat* mat = nullptr;
  if (is_resized) {
    FDASSERT(infer_result.dtype == FDDataType::FP32,
             "Require the data type of output is fp32, but now it's " +
                 Str(const_cast<fastdeploy::FDDataType&>(infer_result.dtype)) +
                 ". Please export PaddleSeg model without argmax and with "
                 "softmax refer to "
                 "https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/"
                 "docs/model_export_cn.md");

    int channel = 1;
    int height = infer_result.shape[1];
    int width = infer_result.shape[2];

    if (result->contain_score_map) {
      channel = infer_result.shape[3];
    }
    cv::Mat temp_mat =
        cv::Mat(height, width, CV_32FC(channel), infer_result.Data());
    auto iter_ipt = (*im_info).find("input_shape");
    FDASSERT(iter_ipt != im_info->end(),
             "Cannot find input_shape from im_info.");
    int ipt_h = iter_ipt->second[0];
    int ipt_w = iter_ipt->second[1];

    mat = new Mat(temp_mat);
    Resize::Run(mat, ipt_w, ipt_h, -1, -1, 1);
    mat->ShareWithTensor(&new_infer_result);
    new_infer_result.shape.insert(new_infer_result.shape.begin(), 1);
    result->shape = new_infer_result.shape;
  } else {
    result->shape = infer_result.shape;
  }
  int out_num =
      std::accumulate(result->shape.begin(), result->shape.begin() + 3, 1,
                      std::multiplies<int>());

  result->shape.erase(result->shape.begin());
  result->label_map.reserve(out_num);
  if (result->contain_score_map) {
    float_t* infer_result_buffer = nullptr;
    if (is_resized) {
      infer_result_buffer = static_cast<float_t*>(new_infer_result.Data());
    } else {
      infer_result_buffer = static_cast<float_t*>(infer_result.Data());
    }
    int64_t height = result->shape[0];
    int64_t width = result->shape[1];
    int64_t num_classes = result->shape[2];
    result->shape.erase(result->shape.begin() + 2);

    result->score_map.reserve(out_num);
    int index = 0;
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        int64_t s = (i * width + j) * num_classes;
        float_t* max_class_score = std::max_element(
            infer_result_buffer + s, infer_result_buffer + s + num_classes);
        int label_id = std::distance(infer_result_buffer + s, max_class_score);
        if (label_id >= 255) {
          FDWARNING << "label_id is stored by uint8_t, now the value is bigger "
                       "than 255, it's "
                    << static_cast<int>(label_id) << "." << std::endl;
        }
        result->label_map[index] = static_cast<uint8_t>(label_id);

        if (with_softmax) {
          double_t total = 0;
          for (int k = 0; k < num_classes; k++) {
            total += exp(*(infer_result_buffer + s + k) - *max_class_score);
          }
          double_t softmax_class_score = 1 / total;
          result->score_map[index] = static_cast<float>(softmax_class_score);

        } else {
          result->score_map[index] = *max_class_score;
        }
        index++;
      }
    }
  } else {
    for (int i = 0; i < out_num; i++) {
      const int64_t* infer_result_buffer =
          reinterpret_cast<const int64_t*>(infer_result.Data());
      result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
    }
  }
  std::free(mat);
  return true;
}

bool Model::Predict(cv::Mat* im, SegmentationResult* result) {
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
  if (!Postprocess(infer_result[0], result, &im_info)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace ppseg
}  // namespace vision
}  // namespace fastdeploy
