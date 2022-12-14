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

#include "fastdeploy/vision/detection/contrib/fastestdet/postprocessor.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace detection {

FastestDetPostprocessor::FastestDetPostprocessor() {
  conf_threshold_ = 0.65;
  nms_threshold_ = 0.45;
  multi_label_ = true;
  max_wh_ = 0.0;
}
float FastestDetPostprocessor::Sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

float FastestDetPostprocessor::Tanh(float x) {
  return 2.0f / (1.0f + exp(-2 * x)) - 1;
}

bool FastestDetPostprocessor::Run(
    const std::vector<FDTensor> &tensors, std::vector<DetectionResult> *results,
    const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info) {
  int batch = 1;

  FDINFO << "view infer output" << std::endl;
  std::vector<FDTensor> debugTensors(tensors);
  FDTensor debugFDT = debugTensors[0];
  debugFDT.PrintInfo("view infer output");

  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {

    (*results)[bs].Clear();
    // output (85,22,22) CHW
    const float* output = reinterpret_cast<const float*>(tensors[0].Data()) + bs * tensors[0].shape[1] * tensors[0].shape[2];
    int output_h = 22; // out map height
    int output_w = 22; // out map weight
    auto iter_out = ims_info[bs].find("output_shape");
    auto iter_ipt = ims_info[bs].find("input_shape");
    FDASSERT(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end(),
             "Cannot find input_shape or output_shape from im_info.");
    float out_h = iter_out->second[0];
    float out_w = iter_out->second[1];
    float ipt_h = iter_ipt->second[0];
    float ipt_w = iter_ipt->second[1];
    float img_width=ipt_w;
    float img_height=ipt_h;

    // handle output boxes from out map
    for (int h = 0; h < output_h; h++) {
      for (int w = 0; w < output_w; w++) {
        // object score
        int obj_score_index = (0 * output_h * output_w) + (h * output_w) + w;
        float obj_score = output[obj_score_index];

        // find max class
        int category = 0;
        float max_score = 0.0f;
        int class_num = 80;
        for (size_t i = 0; i < class_num; i++) {
          int obj_score_index =
              ((5 + i) * output_h * output_w) + (h * output_w) + w;
          float cls_score = output[obj_score_index];
          if (cls_score > max_score) {
            max_score = cls_score;
            category = i;
          }
        }
        float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

        // score threshold
        if (score <= conf_threshold_) {
          continue;
        }
        if (score > conf_threshold_) {
          // handle box x y w h
          int x_offset_index = (1 * output_h * output_w) + (h * output_w) + w;
          int y_offset_index = (2 * output_h * output_w) + (h * output_w) + w;
          int box_width_index = (3 * output_h * output_w) + (h * output_w) + w;
          int box_height_index = (4 * output_h * output_w) + (h * output_w) + w;

          float x_offset = Tanh(output[x_offset_index]);
          float y_offset = Tanh(output[y_offset_index]);
          float box_width = Sigmoid(output[box_width_index]);
          float box_height = Sigmoid(output[box_height_index]);

          float cx = (w + x_offset) / output_w;
          float cy = (h + y_offset) / output_h;

          // convert from [x, y, w, h] to [x1, y1, x2, y2]
          (*results)[bs].boxes.emplace_back(std::array<float, 4>{
            cx - box_width / 2.0f + category * max_wh_,
            cy - box_height / 2.0f + category * max_wh_,
            cx + box_width / 2.0f + category * max_wh_,
            cy + box_height / 2.0f + category * max_wh_});
          (*results)[bs].label_ids.push_back(category);
          (*results)[bs].scores.push_back(score);
        }
      }
    }
    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }

    //utils::NMS(&((*results)[bs]), nms_threshold_);
    // scale boxes to origin shape
    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      int32_t label_id = ((*results)[bs].label_ids)[i];
      // clip box
      (*results)[bs].boxes[i][0] = std::max(((*results)[bs].boxes[i][0]) * ipt_w, 0.0f);
      (*results)[bs].boxes[i][1] = std::max(((*results)[bs].boxes[i][1]) * ipt_h, 0.0f);
      (*results)[bs].boxes[i][2] = std::max(((*results)[bs].boxes[i][2]) * ipt_w, 0.0f);
      (*results)[bs].boxes[i][3] = std::max(((*results)[bs].boxes[i][3]) * ipt_h, 0.0f);
      (*results)[bs].boxes[i][0] = std::min((*results)[bs].boxes[i][0], ipt_w);
      (*results)[bs].boxes[i][1] = std::min((*results)[bs].boxes[i][1], ipt_h);
      (*results)[bs].boxes[i][2] = std::min((*results)[bs].boxes[i][2], ipt_w);
      (*results)[bs].boxes[i][3] = std::min((*results)[bs].boxes[i][3], ipt_h);
    }
    utils::NMS(&((*results)[bs]), nms_threshold_);
  }
  return true;
}

} // namespace detection
} // namespace vision
} // namespace fastdeploy
