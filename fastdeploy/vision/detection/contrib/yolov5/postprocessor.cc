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

#include "fastdeploy/vision/detection/contrib/yolov5/postprocessor.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv5Postprocessor::YOLOv5Postprocessor() {
  conf_threshold_ = 0.25;
  nms_threshold_ = 0.5;
  multi_label_ = true;
  initialized_ = true;
  max_wh_ = 7680.0
}

bool YOLOv5Postprocessor::Postprocess(
    const std::vector<FDTensor>& infer_results, std::vector<DetectionResult>* results,
    const std::map<std::string, std::array<float, 2>>& im_info) {
  auto& infer_result = infer_results[0];
  for (size_t bs = 0; bs < results->size(); ++bs) {
    (*results)[bs].Clear();
    if (multi_label_) {
      (*results)[bs].Reserve(infer_result.shape[1] * (infer_result.shape[2] - 5));
    } else {
      (*results)[bs].Reserve(infer_result.shape[1]);
    }
    if (infer_result.dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    float* data = static_cast<float*>(infer_result.Data()) + bs * infer_result.shape[1] * infer_result.shape[2];
    for (size_t i = 0; i < infer_result.shape[1]; ++i) {
      int s = i * infer_result.shape[2];
      float confidence = data[s + 4];
      if (multi_label_) {
        for (size_t j = 5; j < infer_result.shape[2]; ++j) {
          confidence = data[s + 4];
          float* class_score = data + s + j;
          confidence *= (*class_score);
          // filter boxes by conf_threshold
          if (confidence <= conf_threshold_) {
            continue;
          }
          int32_t label_id = std::distance(data + s + 5, class_score);

          // convert from [x, y, w, h] to [x1, y1, x2, y2]
          (*results)[bs].boxes.emplace_back(std::array<float, 4>{
              data[s] - data[s + 2] / 2.0f + label_id * max_wh_,
              data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
              data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
              data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_});
          (*results)[bs].label_ids.push_back(label_id);
          (*results)[bs].scores.push_back(confidence);
        }
      } else {
        float* max_class_score =
            std::max_element(data + s + 5, data + s + infer_result.shape[2]);
        confidence *= (*max_class_score);
        // filter boxes by conf_threshold
        if (confidence <= conf_threshold_) {
          continue;
        }
        int32_t label_id = std::distance(data + s + 5, max_class_score);
        // convert from [x, y, w, h] to [x1, y1, x2, y2]
        (*results)[bs].boxes.emplace_back(std::array<float, 4>{
            data[s] - data[s + 2] / 2.0f + label_id * max_wh_,
            data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
            data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
            data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_});
        (*results)[bs].label_ids.push_back(label_id);
        (*results)[bs].scores.push_back(confidence);
      }
    }

    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }

    utils::NMS(&((*results)[bs]), nms_threshold_);

    // scale the boxes to the origin image shape
    auto iter_out = im_info.find("output_shape");
    auto iter_ipt = im_info.find("input_shape");
    FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
            "Cannot find input_shape or output_shape from im_info.");
    float out_h = iter_out->second[0];
    float out_w = iter_out->second[1];
    float ipt_h = iter_ipt->second[0];
    float ipt_w = iter_ipt->second[1];
    float scale = std::min(out_h / ipt_h, out_w / ipt_w);
    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      float pad_h = (out_h - ipt_h * scale) / 2;
      float pad_w = (out_w - ipt_w * scale) / 2;
      int32_t label_id = ((*results)[bs].label_ids)[i];
      // clip box
      (*results)[bs].boxes[i][0] = (*results)[bs].boxes[i][0] - max_wh * label_id;
      (*results)[bs].boxes[i][1] = (*results)[bs].boxes[i][1] - max_wh * label_id;
      (*results)[bs].boxes[i][2] = (*results)[bs].boxes[i][2] - max_wh * label_id;
      (*results)[bs].boxes[i][3] = (*results)[bs].boxes[i][3] - max_wh * label_id;
      (*results)[bs].boxes[i][0] = std::max(((*results)[bs].boxes[i][0] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][1] = std::max(((*results)[bs].boxes[i][1] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][2] = std::max(((*results)[bs].boxes[i][2] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][3] = std::max(((*results)[bs].boxes[i][3] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][0] = std::min((*results)[bs].boxes[i][0], ipt_w);
      (*results)[bs].boxes[i][1] = std::min((*results)[bs].boxes[i][1], ipt_h);
      (*results)[bs].boxes[i][2] = std::min((*results)[bs].boxes[i][2], ipt_w);
      (*results)[bs].boxes[i][3] = std::min((*results)[bs].boxes[i][3], ipt_h);
    }
  }
  return true;
}

bool YOLOv5Postprocessor::Run(const std::vector<FDTensor>& tensors, std::vector<DetectionResult>* results,
                              const std::map<std::string, std::array<float, 2>>& im_info) {
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }

  int batch = tensors[0].shape[0];
 
  results->resize(batch);

  if (!Postprocess(tensors, results, im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  return true;
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
