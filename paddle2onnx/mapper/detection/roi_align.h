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
#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class RoiAlignMapper : public Mapper {
 public:
  RoiAlignMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                 int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    MarkAsExperimentalOp();
    GetAttr("pooled_height", &pooled_height_);
    GetAttr("pooled_width", &pooled_width_);
    GetAttr("spatial_scale", &spatial_scale_);
    GetAttr("sampling_ratio", &sampling_ratio_);
    GetAttr("aligned", &aligned_);
  }

  int32_t GetMinOpset(bool verbose = false) {
    Logger(verbose, 10) << RequireOpset(10) << std::endl;
    return 10;
  }
  void Opset10();

 private:
  int64_t pooled_height_;
  int64_t pooled_width_;
  float spatial_scale_;
  int64_t sampling_ratio_;
  bool aligned_;
};

}  // namespace paddle2onnx
