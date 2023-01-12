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

PaddleDetPostprocessor::PaddleDetPostprocessor(const std::string& config_file) {
  this->config_file_ = config_file;
  FDASSERT(ReadPostprocessConfigFromYaml(),
           "Failed to create PaddleDetPostprocessor.");
}

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
    std::cout << "arch: " << arch_ << std::endl;
  } else {
    std::cerr << "Please set model arch,"
              << "support value : YOLO, SSD, RetinaNet, RCNN, Face."
              << std::endl;
    return false;
  }

  if (config["fpn_stride"].IsDefined()) {
    fpn_stride_.clear();
    for (auto item : config["fpn_stride"]) {
      fpn_stride_.emplace_back(item.as<int>());
    }
    printf("[%d,%d,%d,%d]\n", fpn_stride_[0], fpn_stride_[1], fpn_stride_[2],
           fpn_stride_[3]);
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
  const uint8_t* data = reinterpret_cast<const uint8_t*>(tensor.CpuData());
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

      uint8_t* keep_mask_ptr =
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

bool PaddleDetPostprocessor::Run(const std::vector<FDTensor>& tensors,
                                 std::vector<DetectionResult>* results) {
  if (DecodeAndNMSApplied()) {
    return ProcessUnDecodeResults(tensors, results);
  }
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

  // Only detection
  if (tensors.size() <= 2) {
    return true;
  }

  if (tensors[2].Shape()[0] != num_output_boxes) {
    FDERROR << "The first dimension of output mask tensor:"
            << tensors[2].Shape()[0]
            << " is not equal to the first dimension of output boxes tensor:"
            << num_output_boxes << "." << std::endl;
    return false;
  }

  // process for maskrcnn
  return ProcessMask(tensors[2], results);
}

void PaddleDetPostprocessor::ApplyDecodeAndNMS() {
  apply_decode_and_nms_ = true;
}

bool PaddleDetPostprocessor::ProcessUnDecodeResults(
    const std::vector<FDTensor>& tensors,
    std::vector<DetectionResult>* results) {
  FDASSERT(tensors[0].Shape()[0] == 1,
           "ProcessUnDecodeResults only support"
           " input batch = 1.")
  results->resize(1);
  (*results)[0].Resize(0);
  int reg_max = 7;
  int num_class = 80;
  std::vector<const float*> output_data_list_;
  if (arch_ == "PicoDet") {
    for (int i = 0; i < tensors.size(); i++) {
      if (i == 0) {
        num_class = tensors[i].Shape()[2];
      }
      if (i == fpn_stride_.size()) {
        reg_max = tensors[i].Shape()[2] / 4 - 1;
      }
      float* buffer = new float[tensors[i].Numel()];
      memcpy(buffer, tensors[i].Data(), tensors[i].Nbytes());
      output_data_list_.push_back(buffer);
    }
    PicoDetPostProcess(&((*results)[0]), output_data_list_, reg_max, num_class);
  } else {
    FDERROR << "ProcessUnDecodeResults only supported when arch is PicoDet."
            << std::endl;
    return false;
  }
  return true;
}

float FastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

int ActivationFunctionSoftmax(const float* src, float* dst, int length) {
  const float alpha = *std::max_element(src, src + length);
  float denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = FastExp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }

  return 0;
}

void PaddleDetPostprocessor::DisPred2Bbox(
    const float*& dfl_det, int label, float score, int x, int y, int stride,
    int reg_max, fastdeploy::vision::DetectionResult* results) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred{0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float* dis_after_sm = new float[reg_max + 1];
    ActivationFunctionSoftmax(dfl_det + i * (reg_max + 1), dis_after_sm,
                              reg_max + 1);
    for (int j = 0; j < reg_max + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  float xmin = (float)(std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (float)(std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (float)(std::min)(ct_x + dis_pred[2], (float)im_shape_[0]);
  float ymax = (float)(std::min)(ct_y + dis_pred[3], (float)im_shape_[1]);

  results->boxes.emplace_back(std::array<float, 4>{
      xmin / GetScaleFactor()[1], ymin / GetScaleFactor()[0],
      xmax / GetScaleFactor()[1], ymax / GetScaleFactor()[0]});
  results->label_ids.emplace_back(label);
  results->scores.emplace_back(score);
}

void PaddleDetPostprocessor::PicoDetPostProcess(
    fastdeploy::vision::DetectionResult* results,
    std::vector<const float*> outs, int reg_max, int num_class) {
  results->Clear();
  int in_h = im_shape_[0], in_w = im_shape_[1];
  for (int i = 0; i < fpn_stride_.size(); ++i) {
    int feature_h = std::ceil((float)in_h / fpn_stride_[i]);
    int feature_w = std::ceil((float)in_w / fpn_stride_[i]);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      const float* scores = outs[i] + (idx * num_class);
      int row = idx / feature_w;
      int col = idx % feature_w;
      float score = 0;
      int cur_label = 0;
      for (int label = 0; label < num_class; label++) {
        if (scores[label] > score) {
          score = scores[label];
          cur_label = label;
        }
      }
      if (score > score_threshold_) {
        const float* bbox_pred =
            outs[i + fpn_stride_.size()] + (idx * 4 * (reg_max + 1));
        DisPred2Bbox(bbox_pred, cur_label, score, col, row, fpn_stride_[i],
                     reg_max, results);
      }
    }
  }
  fastdeploy::vision::utils::NMS(results, 0.5);
}

std::vector<float> PaddleDetPostprocessor::GetScaleFactor() {
  return scale_factor_;
}

void PaddleDetPostprocessor::SetScaleFactor(float* scale_factor_value) {
  for (int i = 0; i < scale_factor_.size(); ++i) {
    scale_factor_[i] = scale_factor_value[i];
  }
}

bool PaddleDetPostprocessor::DecodeAndNMSApplied() {
  return apply_decode_and_nms_;
}
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
