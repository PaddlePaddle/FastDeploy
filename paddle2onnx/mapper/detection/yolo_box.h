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

#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class YoloBoxMapper : public Mapper {
 public:
  YoloBoxMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    MarkAsExperimentalOp();
    GetAttr("clip_bbox", &clip_bbox_);
    GetAttr("iou_aware", &iou_aware_);
    GetAttr("conf_thresh", &conf_thresh_);
    GetAttr("iou_aware_factor", &iou_aware_factor_);
    GetAttr("class_num", &class_num_);
    GetAttr("downsample_ratio", &downsample_ratio_);
    GetAttr("scale_x_y", &scale_x_y_);
    GetAttr("anchors", &anchors_);
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset11();

 private:
  bool clip_bbox_;
  bool iou_aware_;
  float conf_thresh_;
  float iou_aware_factor_;
  float scale_x_y_;
  int64_t class_num_;
  int64_t downsample_ratio_;
  std::vector<int64_t> anchors_;
};

}  // namespace paddle2onnx
