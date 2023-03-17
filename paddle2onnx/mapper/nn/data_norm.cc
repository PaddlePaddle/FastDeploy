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

#include "paddle2onnx/mapper/nn/data_norm.h"

namespace paddle2onnx {
REGISTER_MAPPER(data_norm, DataNormMapper)

int32_t DataNormMapper::GetMinOpset(bool verbose) {
  if (slot_dim_ > 0) {
    Error() << "slot_dim > 0 is not supported." << std::endl;
    return -1;
  }
  return 7;
}

void DataNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto batch_size_info = GetInput("BatchSize");
  auto batch_sum_info = GetInput("BatchSum");
  auto batch_square_sum_info = GetInput("BatchSquareSum");
  auto output_info = GetOutput("Y");

  Assert(slot_dim_ <= 0, "slot_dim > 0 is not supported.");
  auto mean_arr = helper_->MakeNode("Div", {batch_sum_info[0].name, batch_size_info[0].name})->output(0);
  auto scale_arr = helper_->MakeNode("Div", {batch_size_info[0].name, batch_square_sum_info[0].name})->output(0);
  scale_arr = helper_->MakeNode("Sqrt", {scale_arr})->output(0);
  auto out = helper_->MakeNode("Sub", {input_info[0].name, mean_arr})->output(0);
  helper_->MakeNode("Mul" ,{out, scale_arr}, {output_info[0].name});
}

}  // namespace paddle2onnx
