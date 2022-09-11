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

#include "fastdeploy/vision/detection/ppdet/mask_rcnn.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

namespace fastdeploy {
namespace vision {
namespace detection {

MaskRCNN::MaskRCNN(const std::string& model_file,
                   const std::string& params_file,
                   const std::string& config_file,
                   const RuntimeOption& custom_option,
                   const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool MaskRCNN::Initialize() {
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

bool MaskRCNN::BuildPreprocessPipelineFromConfig() {
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
      FDASSERT(target_size.size() == 2,
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
  processors_.push_back(std::make_shared<HWC2CHW>());
  return true;
}

std::vector<float> MaskRCNN::GetImShapeData(const Mat& mat) {
  float H = static_cast<float>(mat.Height());
  float W = static_cast<float>(mat.Width());
  return {H, W};
}

std::vector<float> MaskRCNN::GetScaleFactorData(const Mat& mat, int origin_h,
                                                int origin_w) {
  float H = static_cast<float>(mat.Height());
  float W = static_cast<float>(mat.Width());
  float h = static_cast<float>(origin_h);
  float w = static_cast<float>(origin_w);
  return {H / h, W / w};
}

bool MaskRCNN::Preprocess(Mat* mat, std::vector<FDTensor>* inputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }
  // im_shape: (1,2), means [H,W] after preprocess.
  // image: (1,3,H,W) input Tensor after preprocess.
  // scale_factor: (1,2), means scale_y and scale_x.
  // ----------------------------------------------
  // +     (1) scale_y = origin_h / input_h       +
  // +     (2) scale_w = origin_w / input_w       +
  // ----------------------------------------------
  int num_inputs_of_runtime = NumInputsOfRuntime();
  inputs->resize(num_inputs_of_runtime);  // 3
  auto im_shape_data = GetImShapeData(*mat);
  auto scale_factor_data = GetScaleFactorData(*mat, origin_h, origin_w);
  for (size_t i = 0; i < num_inputs_of_runtime; ++i) {
    auto input_name = InputInfoOfRuntime(i).name;
    if (input_name == "im_shape") {
      // TODO(qiuyanjun): use new Resize API
      (*inputs)[i].Allocate({1, 2}, FDDataType::FP32, input_name);
      std::memcpy((*inputs)[i].MutableData(), im_shape_data.data(),
                  (*inputs)[i].Nbytes());
    } else if (input_name == "image") {
      (*inputs)[i].name = input_name;
      mat->ShareWithTensor(&((*inputs)[i]));
      (*inputs)[i].ExpandDim(0);  // (1,3,H,W)
    } else if (input_name == "scale_factor") {
      // TODO(qiuyanjun): use new Resize API
      (*inputs)[i].Allocate({1, 2}, FDDataType::FP32, input_name);
      std::memcpy((*inputs)[i].MutableData(), scale_factor_data.data(),
                  (*inputs)[i].Nbytes());
    } else {
      FDERROR << "Input name must be one of (im_shape,image,scale_factor),"
              << "but found (" << input_name << "). "
              << "Please check you model file.\n";
    }
  }
  return true;
}

bool MaskRCNN::Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result) {
  // index 0: bbox_data [N, 6] float32
  // index 1: bbox_num [B=1] int32
  // index 2: mask_data [N, h, w] int32
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");
  FDASSERT(infer_result.size() == 3,
           "The infer_result must contains 3 otuput Tensors, but found %lu",
           infer_result.size());

  FDTensor& box_tensor = infer_result[0];
  FDTensor& box_num_tensor = infer_result[1];
  FDTensor& mask_tensor = infer_result[2];

  int box_num_after_nms = 0;
  if (box_num_tensor.dtype == FDDataType::INT32) {
    box_num_after_nms = *(static_cast<int32_t*>(box_num_tensor.Data()));
  } else if (box_num_tensor.dtype == FDDataType::INT64) {
    box_num_after_nms = *(static_cast<int64_t*>(box_num_tensor.Data()));
  } else {
    FDASSERT(false,
             "The output box_num of PaddleDetection/MaskRCNN model should be "
             "type of int32/int64.");
  }
  if (box_num_after_nms <= 0) {
    return true;  // no object detected.
  }
  // allocate memory
  result->Resize(box_num_after_nms);
  float* box_data = static_cast<float*>(box_tensor.Data());
  for (size_t i = 0; i < box_num_after_nms; ++i) {
    result->label_ids[i] = static_cast<int>(box_data[i * 6]);
    result->scores[i] = box_data[i * 6 + 1];
    result->boxes[i] =
        std::array<float, 4>{box_data[i * 6 + 2], box_data[i * 6 + 3],
                             box_data[i * 6 + 4], box_data[i * 6 + 5]};
  }
  result->contain_masks = true;
  // TODO(qiuyanjun): Cast int64/int8 to int32.
  FDASSERT(mask_tensor.dtype == FDDataType::INT32,
           "The dtype of mask Tensor must be int32 now!");
  // In PaddleDetection/MaskRCNN, the mask_h and mask_w
  // are already aligned with original input image. So,
  // we need to crop it from output mask according to
  // the detected bounding box.
  //   +-----------------------+
  //   |  x1,y1                |
  //   |  +---------------+    |
  //   |  |               |    |
  //   |  |      Crop     |    |
  //   |  |               |    |
  //   |  |               |    |
  //   |  +---------------+    |
  //   |                x2,y2  |
  //   +-----------------------+
  int64_t out_mask_h = mask_tensor.shape[1];
  int64_t out_mask_w = mask_tensor.shape[2];
  int64_t out_mask_numel = out_mask_h * out_mask_w;
  int32_t* out_mask_data = static_cast<int32_t*>(mask_tensor.Data());
  for (size_t i = 0; i < box_num_after_nms; ++i) {
    // crop instance mask according to box
    int64_t x1 = static_cast<int64_t>(result->boxes[i][0]);
    int64_t y1 = static_cast<int64_t>(result->boxes[i][1]);
    int64_t x2 = static_cast<int64_t>(result->boxes[i][2]);
    int64_t y2 = static_cast<int64_t>(result->boxes[i][3]);
    int64_t keep_mask_h = y2 - y1;
    int64_t keep_mask_w = x2 - x1;
    int64_t keep_mask_numel = keep_mask_h * keep_mask_w;
    result->masks[i].Resize(keep_mask_numel);  // int32
    result->masks[i].shape = {keep_mask_h, keep_mask_w};
    int32_t* mask_start_ptr = out_mask_data + i * out_mask_numel;
    int32_t* keep_mask_ptr = static_cast<int32_t*>(result->masks[i].Data());
    for (size_t row = y1; row < y2; ++row) {
      size_t keep_nbytes_in_col = keep_mask_w * sizeof(int32_t);
      int32_t* out_row_start_ptr = mask_start_ptr + row * out_mask_w + x1;
      int32_t* keep_row_start_ptr = keep_mask_ptr + (row - y1) * keep_mask_w;
      std::memcpy(keep_row_start_ptr, out_row_start_ptr, keep_nbytes_in_col);
    }
  }
  return true;
}

bool MaskRCNN::Predict(cv::Mat* im, DetectionResult* result) {
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
