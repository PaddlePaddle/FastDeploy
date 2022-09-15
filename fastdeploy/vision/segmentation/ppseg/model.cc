#include "fastdeploy/vision/segmentation/ppseg/model.h"
#include "fastdeploy/vision.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace segmentation {

PaddleSegModel::PaddleSegModel(const std::string& model_file,
                               const std::string& params_file,
                               const std::string& config_file,
                               const RuntimeOption& custom_option,
                               const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PaddleSegModel::Initialize() {
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

bool PaddleSegModel::BuildPreprocessPipelineFromConfig() {
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
      } else {
        std::string op_name = op["type"].as<std::string>();
        FDERROR << "Unexcepted preprocess operator: " << op_name << "."
                << std::endl;
        return false;
      }
    }
    processors_.push_back(std::make_shared<HWC2CHW>());
  }
  if (cfg["Deploy"]["with_argmax"]) {
    is_with_softmax = cfg["Deploy"]["with_argmax"].as<bool>();
  }
  if (cfg["Deploy"]["with_softmax"]) {
    is_with_argmax = cfg["Deploy"]["with_softmax"].as<bool>();
  }
  return true;
}

bool PaddleSegModel::Preprocess(
    Mat* mat, FDTensor* output,
    std::map<std::string, std::array<int, 2>>* im_info) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (processors_[i]->Name().compare("Resize") == 0) {
      auto processor = dynamic_cast<Resize*>(processors_[i].get());
      int resize_width = -1;
      int resize_height = -1;
      std::tie(resize_width, resize_height) = processor->GetWidthAndHeight();
      if (is_vertical_screen && (resize_width > resize_height)) {
        if (processor->SetWidthAndHeight(resize_height, resize_width)) {
          FDERROR << "Failed to set Resize processor width and height "
                  << processors_[i]->Name() << "." << std::endl;
        }
      }
    }
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

bool PaddleSegModel::Postprocess(
    FDTensor* infer_result, SegmentationResult* result,
    const std::map<std::string, std::array<int, 2>>& im_info) {
  // PaddleSeg has three types of inference output:
  //     1. output with argmax and without softmax. 3-D matrix CHW, Channel
  //     always 1, the element in matrix is classified label_id INT64 Type.
  //     2. output without argmax and without softmax. 4-D matrix NCHW, N always
  //     1, Channel is the num of classes. The element is the logits of classes
  //     FP32
  //     3. output without argmax and with softmax. 4-D matrix NCHW, the result
  //     of 2 with softmax layer
  // Fastdeploy output:
  //     1. label_map
  //     2. score_map(optional)
  //     3. shape: 2-D HW
  FDASSERT(infer_result->dtype == FDDataType::INT64 ||
               infer_result->dtype == FDDataType::FP32,
           "Require the data type of output is int64 or fp32, but now it's %s.",
           Str(infer_result->dtype).c_str());
  result->Clear();

  int num = infer_result->shape[0];
  int channel = infer_result->shape[1];
  int height = infer_result->shape[2];
  int width = infer_result->shape[3];
  int chw = channel * height * width;

  if (is_with_softmax == false && apply_softmax == true) {
    FDTensor softmax_infer_result;
    Softmax(*infer_result, &softmax_infer_result, 1);
    std::memcpy(infer_result->MutableData(), softmax_infer_result.MutableData(),
                num * chw * sizeof(float_t));
  }
  if (is_with_argmax == false) {
    FDASSERT(infer_result->shape[0] == 1, "Only support batch size = 1.");
    // output without argmax
    result->contain_score_map = true;

    std::vector<int64_t> dim{0, 2, 3, 1};
    FDTensor transpose_infer_result;
    Transpose(*infer_result, &transpose_infer_result, dim);
    std::memcpy(infer_result->MutableData(),
                transpose_infer_result.MutableData(),
                num * chw * sizeof(float_t));
    infer_result->shape = {num, height, width, channel};
  }

  // for resize mat below
  FDTensor new_infer_result;
  Mat* mat = nullptr;
  if (is_resized) {
    cv::Mat temp_mat;
    FDTensor2FP32CVMat(&temp_mat, *infer_result, result->contain_score_map);
    // original image shape
    auto iter_ipt = im_info.find("input_shape");
    FDASSERT(iter_ipt != im_info.end(),
             "Cannot find input_shape from im_info.");
    int ipt_h = iter_ipt->second[0];
    int ipt_w = iter_ipt->second[1];

    mat = new Mat(temp_mat);

    Resize::Run(mat, ipt_w, ipt_h, -1, -1, 1);
    mat->ShareWithTensor(&new_infer_result);
    new_infer_result.shape.insert(new_infer_result.shape.begin(), 1);
    result->shape = new_infer_result.shape;
  } else {
    result->shape = infer_result->shape;
  }
  int out_num =
      std::accumulate(result->shape.begin(), result->shape.begin() + 3, 1,
                      std::multiplies<int>());
  // NCHW remove N or CHW remove C
  result->shape.erase(result->shape.begin());
  result->Resize(out_num);
  if (result->contain_score_map) {
    // output with label_map and score_map
    int32_t* argmax_infer_result_buffer = nullptr;
    float_t* score_infer_result_buffer = nullptr;
    FDTensor argmax_infer_result;
    FDTensor max_score_result;
    std::vector<int64_t> reduce_dim{-1};
    // argmax
    if (is_resized) {
      ArgMax(new_infer_result, &argmax_infer_result, -1, FDDataType::INT32);
      Max(new_infer_result, &max_score_result, reduce_dim);
    } else {
      ArgMax(*infer_result, &argmax_infer_result, -1, FDDataType::INT32);
      Max(*infer_result, &max_score_result, reduce_dim);
    }
    argmax_infer_result_buffer =
        static_cast<int32_t*>(argmax_infer_result.Data());
    score_infer_result_buffer = static_cast<float_t*>(max_score_result.Data());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] =
          static_cast<uint8_t>(*(argmax_infer_result_buffer + i));
    }
    std::memcpy(result->score_map.data(), score_infer_result_buffer,
                out_num * sizeof(float_t));
    result->shape.erase(result->shape.begin() + 2);
  } else {
    // output only with label_map
    if (is_resized) {
      float_t* infer_result_buffer =
          static_cast<float_t*>(new_infer_result.Data());
      for (int i = 0; i < out_num; i++) {
        result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
      }
    } else {
      const int64_t* infer_result_buffer =
          reinterpret_cast<const int64_t*>(infer_result->Data());
      for (int i = 0; i < out_num; i++) {
        result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
      }
    }
  }
  delete mat;
  mat = nullptr;
  return true;
}

bool PaddleSegModel::Predict(cv::Mat* im, SegmentationResult* result) {
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
  if (!Postprocess(&infer_result[0], result, im_info)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
