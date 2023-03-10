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

#include "paddle2onnx/mapper/tensor/eye.h"

namespace paddle2onnx {
REGISTER_MAPPER(eye, EyeMapper)

void EyeMapper::ParseValue(const TensorInfo& tensor_info, int64_t* num_val) {
  std::vector<int64_t> value;
  TryGetValue(tensor_info, &value);
  *num_val = value[0];
}

int32_t EyeMapper::GetMinOpset(bool verbose) {
  if (IsAttrVar("num_rows")) {
    if (!IsConstant(GetAttrVar("num_rows")[0])) {
      Error()
          << "While Attribute(num_rows)'s type is Tensor, it's not supported "
             "unless it's a constant tensor."
          << std::endl;
      return -1;
    } else {
      auto info = GetAttrVar("num_rows");
      ParseValue(info[0], &num_rows_);
    }
  } else {
    GetAttr("num_rows", &num_rows_);
  }
  if (IsAttrVar("num_columns")) {
    if (!IsConstant(GetAttrVar("num_columns")[0])) {
      Error() << "While Attribute(num_columns)'s type is Tensor, it's not "
                 "supported "
                 "unless it's a constant tensor."
              << std::endl;
      return -1;
    } else {
      auto info = GetAttrVar("num_columns");
      ParseValue(info[0], &num_columns_);
    }
  } else {
    GetAttr("num_columns", &num_columns_);
  }

  if (num_rows_ <= 0 || num_columns_ <= 0) {
    Error() << "Attribute `num_rows` or  `num_columns` must greater than 0. "
            << std::endl;
    return -1;
  }
  Logger(verbose, 9) << RequireOpset(9) << std::endl;
  return 9;
}

void EyeMapper::Opset9() {
  auto output_info = GetOutput("Out");
  if (IsAttrVar("num_rows")) {
    auto info = GetAttrVar("num_rows");
    ParseValue(info[0], &num_rows_);
  } else {
    GetAttr("num_rows", &num_rows_);
  }
  if (IsAttrVar("num_columns")) {
    auto info = GetAttrVar("num_columns");
    ParseValue(info[0], &num_columns_);
  } else {
    GetAttr("num_columns", &num_columns_);
  }

  std::string constant_node = helper_->Constant(
      {num_rows_, num_columns_}, GetOnnxDtype(output_info[0].dtype), 0);

  auto node =
      helper_->MakeNode("EyeLike", {constant_node}, {output_info[0].name});
}

}  // namespace paddle2onnx
