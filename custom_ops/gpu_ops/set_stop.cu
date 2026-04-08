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

paddle::Tensor GetStop(paddle::Tensor& not_need_stop) {
  bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  auto not_need_stop_cpu =
      GetEmptyTensor({1}, paddle::DataType::BOOL, paddle::CPUPlace());
  bool* not_need_stop_cpu_data =
      const_cast<bool*>(not_need_stop_cpu.data<bool>());
  not_need_stop_cpu_data[0] = not_need_stop_data[0];
  return not_need_stop_cpu;
}

void SetStop(paddle::Tensor& not_need_stop, bool flag) {
  bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = flag;
}
