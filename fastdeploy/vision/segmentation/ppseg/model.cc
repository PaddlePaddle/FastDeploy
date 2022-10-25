// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
                               const ModelFormat& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::ORT, Backend::LITE};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
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
  bool yml_contain_resize_op = false;

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
        yml_contain_resize_op = true;
        const auto& target_size = op["target_size"];
        int resize_width = target_size[0].as<int>();
        int resize_height = target_size[1].as<int>();
        processors_.push_back(
            std::make_shared<Resize>(resize_width, resize_height));
      } else {
        std::string op_name = op["type"].as<std::string>();
        FDERROR << "Unexcepted preprocess operator: " << op_name << "."
                << std::endl;
        return false;
      }
    }
  }
  if (cfg["Deploy"]["input_shape"]) {
    auto input_shape = cfg["Deploy"]["input_shape"];
    int input_batch = input_shape[0].as<int>();
    int input_channel = input_shape[1].as<int>();
    int input_height = input_shape[2].as<int>();
    int input_width = input_shape[3].as<int>();
    if (input_height == -1 || input_width == -1) {
      FDWARNING << "The exported PaddleSeg model is with dynamic shape input, "
	        << "which is not supported by ONNX Runtime and Tensorrt. "
		<< "Only OpenVINO and Paddle Inference are available now. " 
	        << "For using ONNX Runtime or Tensorrt, "
	        << "Please refer to https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export.md"
	        << " to export model with fixed input shape."
	        << std::endl;
      valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::LITE};
      valid_gpu_backends = {Backend::PDINFER};
    }
    if (input_height != -1 && input_width != -1 && !yml_contain_resize_op) {
      processors_.push_back(
          std::make_shared<Resize>(input_width, input_height));
    }
  }
  if (cfg["Deploy"]["output_op"]) {
    std::string output_op = cfg["Deploy"]["output_op"].as<std::string>();
    if (output_op == "softmax") {
      is_with_softmax = true;
      is_with_argmax = false;
    } else if (output_op == "argmax") {
      is_with_softmax = false;
      is_with_argmax = true;
    } else if (output_op == "none") {
      is_with_softmax = false;
      is_with_argmax = false;
    } else {
      FDERROR << "Unexcepted output_op operator in deploy.yml: " << output_op
              << "." << std::endl;
    }
  }
  processors_.push_back(std::make_shared<HWC2CHW>());
  return true;
}

bool PaddleSegModel::Preprocess(Mat* mat, FDTensor* output) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (processors_[i]->Name().compare("Resize") == 0) {
      auto processor = dynamic_cast<Resize*>(processors_[i].get());
      int resize_width = -1;
      int resize_height = -1;
      std::tie(resize_width, resize_height) = processor->GetWidthAndHeight();
      if (is_vertical_screen && (resize_width > resize_height)) {
        if (!(processor->SetWidthAndHeight(resize_height, resize_width))) {
          FDERROR << "Failed to set width and height of "
                  << processors_[i]->Name() << " processor." << std::endl;
        }
      }
    }
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  output->name = InputInfoOfRuntime(0).name;
  return true;
}

bool PaddleSegModel::Postprocess(
    FDTensor* infer_result, SegmentationResult* result,
    const std::map<std::string, std::array<int, 2>>& im_info) {
  // PaddleSeg has three types of inference output:
  //     1. output with argmax and without softmax. 3-D matrix N(C)HW, Channel
  //     always 1, the element in matrix is classified label_id INT64 Type.
  //     2. output without argmax and without softmax. 4-D matrix NCHW, N(batch)
  //     always
  //     1(only support batch size 1), Channel is the num of classes. The
  //     element is the logits of classes
  //     FP32
  //     3. output without argmax and with softmax. 4-D matrix NCHW, the result
  //     of 2 with softmax layer
  // Fastdeploy output:
  //     1. label_map
  //     2. score_map(optional)
  //     3. shape: 2-D HW
  FDASSERT(infer_result->dtype == FDDataType::INT64 ||
               infer_result->dtype == FDDataType::FP32 ||
               infer_result->dtype == FDDataType::INT32,
           "Require the data type of output is int64, fp32 or int32, but now "
           "it's %s.",
           Str(infer_result->dtype).c_str());
  result->Clear();
  FDASSERT(infer_result->shape[0] == 1, "Only support batch size = 1.");

  int64_t infer_batch = infer_result->shape[0];
  int64_t infer_channel = 0;
  int64_t infer_height = 0;
  int64_t infer_width = 0;

  if (is_with_argmax) {
    infer_channel = 1;
    infer_height = infer_result->shape[1];
    infer_width = infer_result->shape[2];
  } else {
    infer_channel = infer_result->shape[1];
    infer_height = infer_result->shape[2];
    infer_width = infer_result->shape[3];
  }
  int64_t infer_chw = infer_channel * infer_height * infer_width;

  bool is_resized = false;
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT(iter_ipt != im_info.end(), "Cannot find input_shape from im_info.");
  int ipt_h = iter_ipt->second[0];
  int ipt_w = iter_ipt->second[1];
  if (ipt_h != infer_height || ipt_w != infer_width) {
    is_resized = true;
  }

  if (!is_with_softmax && apply_softmax) {
    Softmax(*infer_result, infer_result, 1);
  }

  if (!is_with_argmax) {
    // output without argmax
    result->contain_score_map = true;

    std::vector<int64_t> dim{0, 2, 3, 1};
    Transpose(*infer_result, infer_result, dim);
  }
  // batch always 1, so ignore
  infer_result->shape = {infer_height, infer_width, infer_channel};

  // for resize mat below
  FDTensor new_infer_result;
  Mat* mat = nullptr;
  std::vector<float_t>* fp32_result_buffer = nullptr;
  if (is_resized) {
    if (infer_result->dtype == FDDataType::INT64 ||
        infer_result->dtype == FDDataType::INT32) {
      if (infer_result->dtype == FDDataType::INT64) {
        int64_t* infer_result_buffer =
            static_cast<int64_t*>(infer_result->Data());
        // cv::resize don't support `CV_8S` or `CV_32S`
        // refer to https://github.com/opencv/opencv/issues/20991
        // https://github.com/opencv/opencv/issues/7862
        fp32_result_buffer = new std::vector<float_t>(
            infer_result_buffer, infer_result_buffer + infer_chw);
      }
      if (infer_result->dtype == FDDataType::INT32) {
        int32_t* infer_result_buffer =
            static_cast<int32_t*>(infer_result->Data());
        // cv::resize don't support `CV_8S` or `CV_32S`
        // refer to https://github.com/opencv/opencv/issues/20991
        // https://github.com/opencv/opencv/issues/7862
        fp32_result_buffer = new std::vector<float_t>(
            infer_result_buffer, infer_result_buffer + infer_chw);
      }
      infer_result->Resize(infer_result->shape, FDDataType::FP32);
      infer_result->SetExternalData(
          infer_result->shape, FDDataType::FP32,
          static_cast<void*>(fp32_result_buffer->data()));
    }
    mat = new Mat(CreateFromTensor(*infer_result));
    Resize::Run(mat, ipt_w, ipt_h, -1.0f, -1.0f, 1);
    mat->ShareWithTensor(&new_infer_result);
    result->shape = new_infer_result.shape;
  } else {
    result->shape = infer_result->shape;
  }
  // output shape is 2-D HW layout, so out_num = H * W
  int out_num =
      std::accumulate(result->shape.begin(), result->shape.begin() + 2, 1,
                      std::multiplies<int>());
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

  } else {
    // output only with label_map
    if (is_resized) {
      float_t* infer_result_buffer =
          static_cast<float_t*>(new_infer_result.Data());
      for (int i = 0; i < out_num; i++) {
        result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
      }
    } else {
      if (infer_result->dtype == FDDataType::INT64) {
        const int64_t* infer_result_buffer =
            static_cast<const int64_t*>(infer_result->Data());
        for (int i = 0; i < out_num; i++) {
          result->label_map[i] =
              static_cast<uint8_t>(*(infer_result_buffer + i));
        }
      }
      if (infer_result->dtype == FDDataType::INT32) {
        const int32_t* infer_result_buffer =
            static_cast<const int32_t*>(infer_result->Data());
        for (int i = 0; i < out_num; i++) {
          result->label_map[i] =
              static_cast<uint8_t>(*(infer_result_buffer + i));
        }
      }
    }
  }
  // HWC remove C
  result->shape.erase(result->shape.begin() + 2);
  delete fp32_result_buffer;
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
