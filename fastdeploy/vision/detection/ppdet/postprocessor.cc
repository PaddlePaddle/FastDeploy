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

#include "fastdeploy/vision/detection/ppdet/postprocessor.h"

#include "fastdeploy/vision/detection/ppdet/multiclass_nms.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace detection {

bool PaddleDetPostprocessor::ReadPostprocessConfigFromYaml() {
  YAML::Node config;
  try {
    config = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (config["arch"].IsDefined()) {
    arch_ = config["arch"].as<std::string>();
  } else {
    FDERROR << "Please set model arch,"
            << "support value : YOLO, SSD, RetinaNet, RCNN, Face." << std::endl;
    return false;
  }

  if (config["fpn_stride"].IsDefined()) {
    fpn_stride_ = config["fpn_stride"].as<std::vector<int>>();
  }

  if (config["NMS"].IsDefined()) {
    for (const auto& op : config["NMS"]) {
      if (config["background_label"].IsDefined()) {
        multi_class_nms_.background_label =
            op["background_label"].as<int64_t>();
      }
      if (config["keep_top_k"].IsDefined()) {
        multi_class_nms_.keep_top_k = op["keep_top_k"].as<int64_t>();
      }
      if (config["nms_eta"].IsDefined()) {
        multi_class_nms_.nms_eta = op["nms_eta"].as<float>();
      }
      if (config["nms_threshold"].IsDefined()) {
        multi_class_nms_.nms_threshold = op["nms_threshold"].as<float>();
      }
      if (config["nms_top_k"].IsDefined()) {
        multi_class_nms_.nms_top_k = op["nms_top_k"].as<int64_t>();
      }
      if (config["normalized"].IsDefined()) {
        multi_class_nms_.normalized = op["normalized"].as<bool>();
      }
      if (config["score_threshold"].IsDefined()) {
        multi_class_nms_.score_threshold = op["score_threshold"].as<float>();
      }
    }
  }

  if (config["Preprocess"].IsDefined()) {
    for (const auto& op : config["Preprocess"]) {
      std::string op_name = op["type"].as<std::string>();
      if (op_name == "Resize") {
        im_shape_ = op["target_size"].as<std::vector<float>>();
      }
    }
  }

  return true;
}

bool PaddleDetPostprocessor::ProcessMask(
    const FDTensor& tensor, std::vector<DetectionResult>* results) {
  auto shape = tensor.Shape();
  if (tensor.Dtype() != FDDataType::INT32) {
    FDERROR << "The data type of out mask tensor should be INT32, but now it's "
            << tensor.Dtype() << std::endl;
    return false;
  }
  int64_t out_mask_h = shape[1];
  int64_t out_mask_w = shape[2];
  int64_t out_mask_numel = shape[1] * shape[2];
  const auto* data = reinterpret_cast<const uint8_t*>(tensor.CpuData());
  int index = 0;

  for (int i = 0; i < results->size(); ++i) {
    (*results)[i].contain_masks = true;
    (*results)[i].masks.resize((*results)[i].boxes.size());
    for (int j = 0; j < (*results)[i].boxes.size(); ++j) {
      int x1 = static_cast<int>(round((*results)[i].boxes[j][0]));
      int y1 = static_cast<int>(round((*results)[i].boxes[j][1]));
      int x2 = static_cast<int>(round((*results)[i].boxes[j][2]));
      int y2 = static_cast<int>(round((*results)[i].boxes[j][3]));
      int keep_mask_h = y2 - y1;
      int keep_mask_w = x2 - x1;
      int keep_mask_numel = keep_mask_h * keep_mask_w;
      (*results)[i].masks[j].Resize(keep_mask_numel);
      (*results)[i].masks[j].shape = {keep_mask_h, keep_mask_w};
      const uint8_t* current_ptr = data + index * out_mask_numel;

      auto* keep_mask_ptr =
          reinterpret_cast<uint8_t*>((*results)[i].masks[j].Data());
      for (int row = y1; row < y2; ++row) {
        size_t keep_nbytes_in_col = keep_mask_w * sizeof(uint8_t);
        const uint8_t* out_row_start_ptr = current_ptr + row * out_mask_w + x1;
        uint8_t* keep_row_start_ptr = keep_mask_ptr + (row - y1) * keep_mask_w;
        std::memcpy(keep_row_start_ptr, out_row_start_ptr, keep_nbytes_in_col);
      }
      index += 1;
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessWithNMS(
    const std::vector<FDTensor>& tensors,
    std::vector<DetectionResult>* results) {
  // Get number of boxes for each input image
  std::vector<int> num_boxes(tensors[1].shape[0]);
  int total_num_boxes = 0;
  if (tensors[1].dtype == FDDataType::INT32) {
    const auto* data = static_cast<const int32_t*>(tensors[1].CpuData());
    for (size_t i = 0; i < tensors[1].shape[0]; ++i) {
      num_boxes[i] = static_cast<int>(data[i]);
      total_num_boxes += num_boxes[i];
    }
  } else if (tensors[1].dtype == FDDataType::INT64) {
    const auto* data = static_cast<const int64_t*>(tensors[1].CpuData());
    for (size_t i = 0; i < tensors[1].shape[0]; ++i) {
      num_boxes[i] = static_cast<int>(data[i]);
      total_num_boxes += num_boxes[i];
    }
  }

  // Special case for TensorRT, it has fixed output shape of NMS
  // So there's invalid boxes in its' output boxes
  int num_output_boxes = static_cast<int>(tensors[0].Shape()[0]);
  bool contain_invalid_boxes = false;
  if (total_num_boxes != num_output_boxes) {
    if (num_output_boxes % num_boxes.size() == 0) {
      contain_invalid_boxes = true;
    } else {
      FDERROR << "Cannot handle the output data for this model, unexpected "
                 "situation."
              << std::endl;
      return false;
    }
  }

  // Get boxes for each input image
  results->resize(num_boxes.size());

  if (tensors[0].shape[0] == 0) {
    // No detected boxes
    return true;
  }

  const auto* box_data = static_cast<const float*>(tensors[0].CpuData());
  int offset = 0;
  for (size_t i = 0; i < num_boxes.size(); ++i) {
    const float* ptr = box_data + offset;
    (*results)[i].Reserve(num_boxes[i]);
    for (size_t j = 0; j < num_boxes[i]; ++j) {
      (*results)[i].label_ids.push_back(
          static_cast<int32_t>(round(ptr[j * 6])));
      (*results)[i].scores.push_back(ptr[j * 6 + 1]);
      (*results)[i].boxes.emplace_back(std::array<float, 4>(
          {ptr[j * 6 + 2], ptr[j * 6 + 3], ptr[j * 6 + 4], ptr[j * 6 + 5]}));
    }
    if (contain_invalid_boxes) {
      offset += static_cast<int>(num_output_boxes * 6 / num_boxes.size());
    } else {
      offset += static_cast<int>(num_boxes[i] * 6);
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessWithoutNMS(
    const std::vector<FDTensor>& tensors,
    std::vector<DetectionResult>* results) {
  int boxes_index = 0;
  int scores_index = 1;

  // Judge the index of the input Tensor
  if (tensors[0].shape[1] == tensors[1].shape[2]) {
    boxes_index = 0;
    scores_index = 1;
  } else if (tensors[0].shape[2] == tensors[1].shape[1]) {
    boxes_index = 1;
    scores_index = 0;
  } else {
    FDERROR << "The shape of boxes and scores should be [batch, boxes_num, "
               "4], [batch, classes_num, boxes_num]"
            << std::endl;
    return false;
  }

  // do multi class nms
  multi_class_nms_.Compute(
      static_cast<const float*>(tensors[boxes_index].Data()),
      static_cast<const float*>(tensors[scores_index].Data()),
      tensors[boxes_index].shape, tensors[scores_index].shape);
  auto num_boxes = multi_class_nms_.out_num_rois_data;
  auto box_data =
      static_cast<const float*>(multi_class_nms_.out_box_data.data());

  // Get boxes for each input image
  results->resize(num_boxes.size());
  int offset = 0;
  for (size_t i = 0; i < num_boxes.size(); ++i) {
    const float* ptr = box_data + offset;
    (*results)[i].Reserve(num_boxes[i]);
    for (size_t j = 0; j < num_boxes[i]; ++j) {
      (*results)[i].label_ids.push_back(
          static_cast<int32_t>(round(ptr[j * 6])));
      (*results)[i].scores.push_back(ptr[j * 6 + 1]);
      (*results)[i].boxes.emplace_back(std::array<float, 4>(
          {ptr[j * 6 + 2], ptr[j * 6 + 3], ptr[j * 6 + 4], ptr[j * 6 + 5]}));
    }
    offset += (num_boxes[i] * 6);
  }

  // do scale
  if (GetScaleFactor()[0] != 0) {
    for (auto& result : *results) {
      for (auto& box : result.boxes) {
        box[0] /= GetScaleFactor()[1];
        box[1] /= GetScaleFactor()[0];
        box[2] /= GetScaleFactor()[1];
        box[3] /= GetScaleFactor()[0];
      }
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessGeneral(
    const std::vector<FDTensor>& tensors,
    std::vector<DetectionResult>* results) {
  // Do process according to whether NMS exists.
  if (with_nms_) {
    if (!ProcessWithNMS(tensors, results)) {
      return false;
    }
  } else {
    if (!ProcessWithoutNMS(tensors, results)) {
      return false;
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessSolov2(
    const std::vector<FDTensor>& tensors,
    std::vector<DetectionResult>* results) {
  if (tensors.size() != 4) {
    FDERROR << "The size of tensors for solov2 must be 4." << std::endl;
    return false;
  }

  if () return true;
}

bool PaddleDetPostprocessor::Run(const std::vector<FDTensor>& tensors,
                                 std::vector<DetectionResult>* results) {
  // For Solov2
  if (arch_ == "SOLOv2") {
    return ProcessSolov2(tensors, results);
  }

  // Only detection
  if (tensors.size() <= 2) {
    return ProcessGeneral(tensors, results);
  }

  // process for maskrcnn
  ProcessGeneral(tensors, results);
  if (tensors[2].Shape()[0] != tensors[0].Shape()[0]) {
    FDERROR << "The first dimension of output mask tensor:"
            << tensors[2].Shape()[0]
            << " is not equal to the first dimension of output boxes tensor:"
            << tensors[0].Shape()[0] << "." << std::endl;
    return false;
  }
  return ProcessMask(tensors[2], results);
}
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
