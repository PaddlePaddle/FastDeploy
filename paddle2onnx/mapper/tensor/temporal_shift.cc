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

#include "paddle2onnx/mapper/tensor/temporal_shift.h"

namespace paddle2onnx {
REGISTER_MAPPER(temporal_shift, TemporalShiftMapper)

int32_t TemporalShiftMapper::GetMinOpset(bool verbose) {
  if (data_format_ == "NHWC") {
    Error() << "Only support data_format of NCHW, but now the data format is "
            << data_format_ << "." << std::endl;
    return -1;
  }
  auto input_info = GetOutput("Out");
  if (input_info[0].Rank() != 4) {
    Error() << "The input dims must be 4, but now the input dims is "
            << std::to_string(input_info[0].Rank()) << "." << std::endl;
    return -1;
  }
  return 7;
}

void TemporalShiftMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  int64_t C = input_info[0].shape[1];
  int64_t H = input_info[0].shape[2];
  int64_t W = input_info[0].shape[3];
  std::vector<int64_t> reshape_shape = {-1, seg_num_, C, H, W};

  std::string reshape_input =
      helper_->Reshape(input_info[0].name, reshape_shape);

  std::vector<int64_t> paddings(10, 0);
  paddings[1] = 1;
  paddings[6] = 1;

  std::string padding_constant_node =
      helper_->Constant(GetOnnxDtype(P2ODataType::INT64), paddings);

  std::string pad_node = "";
  if (helper_->GetOpsetVersion() < 11) {
    auto node = helper_->MakeNode("Pad", {reshape_input});
    AddAttribute(node, "pads", paddings);
    float val = 0.0;
    AddAttribute(node, "value", val);
    pad_node = node->output(0);
  } else {
    auto node =
        helper_->MakeNode("Pad", {reshape_input, padding_constant_node});
    pad_node = node->output(0);
  }

  int64_t C1 = C * shift_ratio_;
  int64_t C2 = 2 * C * shift_ratio_;
  std::string slice_1 =
      helper_->Slice(pad_node, {1, 2}, {0, 0}, {seg_num_, C1});
  std::string slice_2 =
      helper_->Slice(pad_node, {1, 2}, {2, C1}, {2 + seg_num_, C2});
  std::string slice_3 =
      helper_->Slice(pad_node, {1, 2}, {1, C2}, {1 + seg_num_, C});
  std::string concat_out = helper_->Concat({slice_1, slice_2, slice_3}, 2);
  helper_->Reshape(concat_out, output_info[0].name, {-1, C, H, W});
}

}  // namespace paddle2onnx
