// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"

paddle::Tensor ComputeTokenNum(const paddle::Tensor& seq_len_this_time_cpu) {
  const int* seq_len_data = seq_len_this_time_cpu.data<int>();

  auto token_num =
      GetEmptyTensor({1}, paddle::DataType::INT32, paddle::CPUPlace());
  int* token_num_data = token_num.data<int>();

  int num = 0;
  for (int i = 0; i < seq_len_this_time_cpu.numel(); ++i) {
    int v = seq_len_data[i];
    if (v > 0) {
      num += v;
    }
  }

  token_num_data[0] = num;
  return token_num;
}
