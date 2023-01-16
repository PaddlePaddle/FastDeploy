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

#include "fastdeploy/vision/detection/ppdet/multiclass_nms.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
namespace fastdeploy {
namespace vision {
namespace detection {
PPDetDecode::PPDetDecode(const std::string& config_file) {
  config_file_ = config_file;
  ReadPostprocessConfigFromYaml();
}

/***************************************************************
 *  @name       ReadPostprocessConfigFromYaml
 *  @brief      Read decode config from yaml.
 *  @note       read arch
 *              read fpn_stride
 *              read nms_threshold on NMS
 *              read score_threshold on NMS
 *              read target_size
 ***************************************************************/
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
    fpn_stride_ = config["fpn_stride"].as<std::vector<int>>();
  }

  if (config["NMS"].IsDefined()) {
    for (const auto& op : config["NMS"]) {
      if (config["nms_threshold"].IsDefined()) {
        nms_threshold_ = op["nms_threshold"].as<float>();
      } else if (config["score_threshold"].IsDefined()) {
        score_threshold_ = op["score_threshold"].as<float>();
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

/***************************************************************
 *  @name       DecodeAndNMS
 *  @brief      Read batch and call different decode functions.
 *  @param      tensors: model output tensor
 *              results: detection results
 *  @note       Only support arch is Picodet.
 ***************************************************************/
bool PPDetDecode::DecodeAndNMS(const std::vector<FDTensor>& tensors,
                               std::vector<DetectionResult>* results) {
  if (tensors.size() == 2) {
    int boxes_index = 0;
    int scores_index = 1;
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

    PaddleMultiClassNMS nms;
    nms.background_label = -1;
    nms.keep_top_k = 100;
    nms.nms_eta = 1.0;
    nms.nms_threshold = 0.5;
    nms.score_threshold = 0.3;
    nms.nms_top_k = 1000;
    nms.normalized = true;
    nms.Compute(static_cast<const float*>(tensors[boxes_index].Data()),
                static_cast<const float*>(tensors[scores_index].Data()),
                tensors[boxes_index].shape, tensors[scores_index].shape);

    auto num_boxes = nms.out_num_rois_data;
    auto box_data = static_cast<const float*>(nms.out_box_data.data());
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
    return true;
  } else {
    FDASSERT(tensors.size() == fpn_stride_.size() * 2,
             "The size of output must be fpn_stride * 2.")
    batchs_ = static_cast<int>(tensors[0].shape[0]);
    if (arch_ == "PicoDet") {
      for (int i = 0; i < tensors.size(); i++) {
        if (i == 0) {
          num_class_ = static_cast<int>(tensors[i].Shape()[2]);
        }
        if (i == fpn_stride_.size()) {
          reg_max_ = static_cast<int>(tensors[i].Shape()[2] / 4);
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
}

/***************************************************************
 *  @name       PicoDetPostProcess
 *  @brief      Do decode and NMS for Picodet.
 *  @param      outs: model output tensor
 *              results: detection results
 *  @note       Only support PPYOLOE and Picodet.
 ***************************************************************/
bool PPDetDecode::PicoDetPostProcess(const std::vector<FDTensor>& outs,
                                     std::vector<DetectionResult>* results) {
  for (int batch = 0; batch < batchs_; ++batch) {
    auto& result = (*results)[batch];
    result.Clear();
    for (int i = batch * batchs_ * fpn_stride_.size();
         i < fpn_stride_.size() * (batch + 1); ++i) {
      int feature_h =
          std::ceil(im_shape_[0] / static_cast<float>(fpn_stride_[i]));
      int feature_w =
          std::ceil(im_shape_[1] / static_cast<float>(fpn_stride_[i]));
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
    fastdeploy::vision::utils::NMS(&result, nms_threshold_);
  }
  return results;
}

/***************************************************************
 *  @name       FastExp
 *  @brief      Do exp op
 *  @param      x: input data
 *  @return     float
 ***************************************************************/
float FastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

/***************************************************************
 *  @name       ActivationFunctionSoftmax
 *  @brief      Do Softmax with reg_max.
 *  @param      src: input data
 *              dst: output data
 *  @return     float
 ***************************************************************/
int PPDetDecode::ActivationFunctionSoftmax(const float* src, float* dst) {
  const float alpha = *std::max_element(src, src + reg_max_);
  float denominator{0};

  for (int i = 0; i < reg_max_; ++i) {
    dst[i] = FastExp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < reg_max_; ++i) {
    dst[i] /= denominator;
  }

  return 0;
}

/***************************************************************
 *  @name       DisPred2Bbox
 *  @brief      Do Decode.
 *  @param      dfl_det: detection data
 *              label: label id
 *              score: confidence
 *              x: col
 *              y: row
 *              stride: stride
 *              results: detection results
 ***************************************************************/
void PPDetDecode::DisPred2Bbox(const float*& dfl_det, int label, float score,
                               int x, int y, int stride,
                               fastdeploy::vision::DetectionResult* results) {
  float ct_x = static_cast<float>(x + 0.5) * static_cast<float>(stride);
  float ct_y = static_cast<float>(y + 0.5) * static_cast<float>(stride);
  std::vector<float> dis_pred{0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    auto* dis_after_sm = new float[reg_max_];
    ActivationFunctionSoftmax(dfl_det + i * (reg_max_), dis_after_sm);
    for (int j = 0; j < reg_max_; j++) {
      dis += static_cast<float>(j) * dis_after_sm[j];
    }
    dis *= static_cast<float>(stride);
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
