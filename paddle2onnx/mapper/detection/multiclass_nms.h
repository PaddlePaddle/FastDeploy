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

class NMSMapper : public Mapper {
 public:
  NMSMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
            int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    // NMS is a post process operators for object detection
    // We have found there're difference between `multi_class_nms3` in
    // PaddlePaddle and `NonMaxSuppresion` in ONNX
    MarkAsExperimentalOp();
    GetAttr("normalized", &normalized_);
    GetAttr("nms_threshold", &nms_threshold_);
    GetAttr("score_threshold", &score_threshold_);
    GetAttr("nms_eta", &nms_eta_);
    // The `nms_top_k` in Paddle and `max_output_boxes_per_class` in ONNX share
    // the same meaning But the filter process may not be same Since NMS is just
    // a post process for Detection, we are not going to export it with exactly
    // same result. We will make a precision performance in COCO or Pascal VOC
    // data later.
    GetAttr("nms_top_k", &nms_top_k_);
    GetAttr("background_label", &background_label_);
    GetAttr("keep_top_k", &keep_top_k_);
  }

  int32_t GetMinOpset(bool verbose = false);
  void KeepTopK(const std::string& selected_indices);
  void Opset10();
  void ExportForTensorRT();
  void ExportAsCustomOp();

 private:
  bool normalized_;
  float nms_threshold_;
  float score_threshold_;
  float nms_eta_;
  int64_t nms_top_k_;
  int64_t background_label_;
  int64_t keep_top_k_;
};

}  // namespace paddle2onnx
