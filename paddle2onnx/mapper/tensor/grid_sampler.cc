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

#include "paddle2onnx/mapper/tensor/grid_sampler.h"

namespace paddle2onnx {
REGISTER_MAPPER(grid_sampler, GridSamplerMapper)

int32_t GridSamplerMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 16) << RequireOpset(16) << std::endl;
  return 16;
};

void GridSamplerMapper::Opset16() {
  auto x_info = GetInput("X");
  auto grid_info = GetInput("Grid");
  auto out_info = GetOutput("Output");
  std::string cast_input =
      helper_->AutoCast(x_info[0].name, x_info[0].dtype, P2ODataType::FP32);
  std::string cast_grid = helper_->AutoCast(
      grid_info[0].name, grid_info[0].dtype, P2ODataType::FP32);
  auto node = helper_->MakeNode("GridSample", {cast_input, cast_grid});
  AddAttribute(node, "padding_mode", padding_mode_);
  AddAttribute(node, "mode", mode_);
  AddAttribute(node, "align_corners", static_cast<int64_t>(align_corners_));
  helper_->AutoCast(node->output(0), out_info[0].name, P2ODataType::FP32,
                    out_info[0].dtype);
}

}  // namespace paddle2onnx
