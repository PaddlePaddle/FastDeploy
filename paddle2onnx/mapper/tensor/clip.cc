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

#include "paddle2onnx/mapper/tensor/clip.h"

namespace paddle2onnx {
REGISTER_MAPPER(clip, ClipMapper)

int32_t ClipMapper::GetMinOpset(bool verbose) {
  bool has_max_tensor_input = HasInput("Max");
  bool has_min_tensor_input = HasInput("Min");
  if (has_max_tensor_input || has_min_tensor_input) {
    return 11;
  }
  return 7;
}

void ClipMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  bool has_max_tensor_input = HasInput("Max");
  bool has_min_tensor_input = HasInput("Min");

  if (has_max_tensor_input || has_min_tensor_input) {
    bool dtype_converted = false;
    std::string input_name = input_info[0].name;
    int32_t dtype = input_info[0].dtype;
    // onnxruntime only supports float input
    if (input_info[0].dtype != P2ODataType::FP32) {
      input_name = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                     P2ODataType::FP32);
      dtype_converted = true;
      dtype = P2ODataType::FP32;
    }
    std::string max_name;
    if (has_max_tensor_input) {
      auto max_info = GetInput("Max");
      max_name = helper_->AutoCast(max_info[0].name, max_info[0].dtype, dtype);
      if (max_info[0].Rank() > 0) {
        max_name = helper_->Squeeze(max_name, {});
      }
    } else {
      float max_val;
      GetAttr("max", &max_val);
      max_name = helper_->Constant({}, GetOnnxDtype(dtype), max_val);
    }
    std::string min_name;
    if (has_min_tensor_input) {
      auto min_info = GetInput("Min");
      min_name = helper_->AutoCast(min_info[0].name, min_info[0].dtype, dtype);
      if (min_info[0].Rank() > 0) {
        min_name = helper_->Squeeze(min_name, {});
      }
    } else {
      float min_val;
      GetAttr("min", &min_val);
      min_name = helper_->Constant({}, GetOnnxDtype(dtype), min_val);
    }
    if (dtype_converted) {
      auto node = helper_->MakeNode("Clip", {input_name, min_name, max_name});
      helper_->AutoCast(node->output(0), output_info[0].name, P2ODataType::FP32,
                        output_info[0].dtype);
    } else {
      helper_->MakeNode("Clip", {input_name, min_name, max_name},
                        {output_info[0].name});
    }
  } else {
    float max_val;
    GetAttr("max", &max_val);
    float min_val;
    GetAttr("min", &min_val);
    helper_->Clip(input_info[0].name, output_info[0].name, min_val, max_val,
                  input_info[0].dtype);
  }
}

}  // namespace paddle2onnx
