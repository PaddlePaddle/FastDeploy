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

#include "paddle2onnx/mapper/tensor/assign_value.h"

#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(assign_value, AssignValueMapper)

int32_t AssignValueMapper::GetMinOpset(bool verbose) {
  int32_t dtype = static_cast<int32_t>(dtype_);
  if (dtype != P2ODataType::INT32 && dtype != P2ODataType::INT64 &&
      dtype != P2ODataType::FP32) {
    Error() << "Only supports int32/int64/float32." << std::endl;
    return -1;
  }
  return 7;
}

void AssignValueMapper::Opset7() {
  auto output_info = GetOutput("Out");
  int32_t dtype = static_cast<int32_t>(dtype_);
  if (dtype == P2ODataType::INT32) {
    helper_->Assign(output_info[0].name, GetOnnxDtype(output_info[0].dtype),
                    shape_, int64_values_);
  } else if (dtype == P2ODataType::FP32) {
    helper_->Assign(output_info[0].name, GetOnnxDtype(output_info[0].dtype),
                    shape_, fp32_values_);
  } else if (dtype == P2ODataType::INT64) {
    helper_->Assign(output_info[0].name, GetOnnxDtype(output_info[0].dtype),
                    shape_, int64_values_);
  }
}

}  // namespace paddle2onnx
