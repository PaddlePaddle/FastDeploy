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

#include "paddle2onnx/mapper/nn/conv3d.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(conv3d, Conv3dMapper)

int32_t Conv3dMapper::GetMinOpset(bool verbose) {
  // NDHWC is not supported
  if (data_format_ == "NDHWC") {
    Error() << "Cannot support input with NDHWC format." << std::endl;
    return -1;
  }
  if (padding_algorithm_ == "EXPLICIT") {
    if (paddings_.size() != 3 && paddings_.size() != 6) {
      Error() << "While padding_algorithm is EXPLICIT, size of paddings should "
                 "be 3 or 6."
              << std::endl;
      return -1;
    }
  }
  if (dilations_[0] != 1 || dilations_[1] != 1 || dilations_[2] != 1) {
    if (padding_algorithm_ == "SAME") {
      Error() << "While dilations != 1, cannot support padding = 'SAME'."
              << std::endl;
      return -1;
    }
  }
  return 7;
}

void Conv3dMapper::Opset7() {
  auto kernel_info = GetInput("Filter");
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Output");

  auto node = helper_->MakeNode(
      "Conv", {input_info[0].name, kernel_info[0].name}, {output_info[0].name});
  AddAttribute(node, "dilations", dilations_);
  std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                       kernel_info[0].shape[3],
                                       kernel_info[0].shape[4]};
  AddAttribute(node, "kernel_shape", kernel_shape);
  AddAttribute(node, "strides", strides_);
  AddAttribute(node, "group", groups_);
  if (padding_algorithm_ == "SAME") {
    std::string auto_pad = "SAME_UPPER";
    AddAttribute(node, "auto_pad", auto_pad);
  } else if (padding_algorithm_ == "VALID") {
    std::string auto_pad = "VALID";
    AddAttribute(node, "auto_pad", auto_pad);
  } else {
    std::vector<int64_t> paddings;
    if (paddings_.size() == 3) {
      paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
      paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
    } else {
      std::vector<int64_t> index = {0, 2, 4, 1, 3, 5};
      for (auto &i : index) {
        paddings.push_back(paddings_[i]);
      }
    }
    AddAttribute(node, "pads", paddings);
  }
}

}  // namespace paddle2onnx
