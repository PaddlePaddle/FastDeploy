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
#include "paddle2onnx/mapper/tensor/fill_constant_batch_size_like.h"

namespace paddle2onnx {

REGISTER_MAPPER(fill_constant_batch_size_like, FillConstantBatchSizeLikeMapper)

int32_t FillConstantBatchSizeLikeMapper::GetMinOpset(bool verbose) {
  auto out_info = GetOutput("Out");
  if (out_info[0].dtype == P2ODataType::BOOL) {
    Error() << "Dtype of boolean is not supported." << std::endl;
    return -1;
  }
  return 7;
}

void FillConstantBatchSizeLikeMapper::Opset7() {
  auto input_info = GetInput("Input");
  auto out_info = GetOutput("Out");
  float value = value_;
  if (!str_value_.empty()) {
    std::stringstream convert_stream(str_value_);
    convert_stream >> value;
  }
  std::vector<int64_t> shape;
  shape.assign(shape_.begin(), shape_.end());
  shape[output_dim_idx_] = 1;

  auto input_shape =
      helper_->MakeNode("Shape", {input_info[0].name})->output(0);
  auto batch =
      helper_->Slice(input_shape, {0}, {input_dim_idx_}, {input_dim_idx_ + 1});

  if (output_dim_idx_ == 0) {
    auto constant =
        helper_->Constant(shape, GetOnnxDtype(out_info[0].dtype), value);
    auto repeat = batch;
    if (shape.size() > 1) {
      auto tmp = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                   std::vector<int64_t>(shape.size() - 1, 1));
      repeat = helper_->Concat({batch, tmp}, 0);
    }
    helper_->MakeNode("Tile", {constant, repeat}, {out_info[0].name});
  } else if (output_dim_idx_ == shape.size() - 1) {
    auto constant =
        helper_->Constant(shape, GetOnnxDtype(out_info[0].dtype), value);
    auto repeat = batch;
    if (shape.size() > 1) {
      auto tmp = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                   std::vector<int64_t>(shape.size() - 1, 1));
      repeat = helper_->Concat({tmp, batch}, 0);
    }
    helper_->MakeNode("Tile", {constant, repeat}, {out_info[0].name});
  } else {
    shape.erase(shape.begin() + output_dim_idx_);
    shape.insert(shape.begin(), int64_t(1));
    auto constant =
        helper_->Constant(shape, GetOnnxDtype(out_info[0].dtype), value);
    auto repeat = batch;
    if (shape.size() > 1) {
      auto tmp = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                   std::vector<int64_t>(shape.size() - 1, 1));
      repeat = helper_->Concat({batch, tmp}, 0);
    }
    auto out = helper_->MakeNode("Tile", {constant, repeat})->output(0);
    auto perm = Arange(1, shape.size());
    perm.insert(perm.begin() + output_dim_idx_, int64_t(0));
    helper_->Transpose(out, out_info[0].name, perm);
  }
}

}  // namespace paddle2onnx
