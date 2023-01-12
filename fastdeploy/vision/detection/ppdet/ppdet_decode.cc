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
#include "ppdet_decode.h"

#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
namespace fastdeploy {
namespace vision {
namespace detection {
PPDetDecode::PPDetDecode(const std::string& config_file) {
  config_file_ = config_file;
  ReadPostprocessConfigFromYaml();
}

bool PPDetDecode::ReadPostprocessConfigFromYaml() {
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
    std::cerr << "Please set model arch,"
              << "support value : YOLO, SSD, RetinaNet, RCNN, Face."
              << std::endl;
    return false;
  }

  if (config["fpn_stride"].IsDefined()) {
    fpn_stride_.clear();
    for (auto item : config["fpn_stride"]) {
      fpn_stride_.emplace_back(item.as<float>());
    }
  }
  return true;
}

bool PPDetDecode::DecodeAndNMS(const std::vector<FDTensor>& tensors,
                               std::vector<DetectionResult>* results) {
  batchs_ = tensors[0].shape[0];
  if (arch_ == "PicoDet") {
    for (int i = 0; i < tensors.size(); i++) {
      if (i == 0) {
        num_class_ = tensors[i].Shape()[2];
      }
      if (i == fpn_stride_.size()) {
        reg_max_ = tensors[i].Shape()[2] / 4;
      }
    }
    for (int i = 0; i < results->size(); ++i) {
      PicoDetPostProcess(tensors, results);
    }
  } else {
    FDERROR << "ProcessUnDecodeResults only supported when arch is PicoDet."
            << std::endl;
    return false;
  }
  return true;
}

bool PPDetDecode::PicoDetPostProcess(const std::vector<FDTensor>& outs,
                                     std::vector<DetectionResult>* results) {
  for (int batch = 0; batch < batchs_; ++batch) {
    auto& result = (*results)[batch];
    result.Clear();
    for (int i = batch * batchs_ * fpn_stride_.size();
         i < fpn_stride_.size() * (batch + 1); ++i) {
      int feature_h = std::ceil(im_shape_[0] / fpn_stride_[i]);
      int feature_w = std::ceil(im_shape_[1] / fpn_stride_[i]);
      for (int idx = 0; idx < feature_h * feature_w; idx++) {
        const auto* scores =
            static_cast<const float*>(outs[i].Data()) + (idx * num_class_);
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class_; label++) {
          if (scores[label] > score) {
            score = scores[label];
            cur_label = label;
          }
        }
        if (score > score_threshold_) {
          const auto* bbox_pred =
              static_cast<const float*>(outs[i + fpn_stride_.size()].Data()) +
              (idx * 4 * (reg_max_));
          DisPred2Bbox(bbox_pred, cur_label, score, col, row, fpn_stride_[i],
                       &result);
        }
      }
    }
    fastdeploy::vision::utils::NMS(&result, 0.5);
  }
  return results;
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

void PPDetDecode::DisPred2Bbox(const float*& dfl_det, int label, float score,
                               int x, int y, int stride,
                               fastdeploy::vision::DetectionResult* results) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred{0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float* dis_after_sm = new float[reg_max_];
    ActivationFunctionSoftmax(dfl_det + i * (reg_max_), dis_after_sm, reg_max_);
    for (int j = 0; j < reg_max_; j++) {
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

  results->boxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
  results->label_ids.emplace_back(label);
  results->scores.emplace_back(score);
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
