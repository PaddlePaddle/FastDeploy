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

class InterpolateMapper : public Mapper {
 public:
  InterpolateMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("data_layout", &data_layout_);
    GetAttr("align_corners", &align_corners_);
    GetAttr("align_mode", &align_mode_);
    GetAttr("out_d", &out_d_);
    GetAttr("out_h", &out_h_);
    GetAttr("out_w", &out_w_);
    method_ = OpType();

    resize_mapper_["bilinear_interp"] = "linear";
    resize_mapper_["bilinear_interp_v2"] = "linear";
    resize_mapper_["nearest_interp_v2"] = "nearest";
    resize_mapper_["bicubic_interp_v2"] = "cubic";
    resize_mapper_["linear_interp_v2"] = "linear";
    resize_mapper_["trilinear_interp_v2"] = "linear";
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset11();

 private:
  std::string ComputeOutSize();
  std::string ComputeScale();
  std::map<std::string, std::string> resize_mapper_;
  std::string method_;
  std::string data_layout_;
  int64_t align_mode_;
  int64_t out_d_;
  int64_t out_h_;
  int64_t out_w_;
  bool align_corners_;
};

}  // namespace paddle2onnx
