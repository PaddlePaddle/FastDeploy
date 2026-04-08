// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"

paddle::Tensor prepack_B(
    paddle::Tensor const& B, paddle::DataType const& a_type, int64_t b_type_id,
    std::string const& maybe_group_scales_type_str) {
  machete::ScalarType const b_type = machete::ScalarType::from_id(b_type_id);
  std::optional<paddle::DataType> maybe_group_scales_type;
  if (maybe_group_scales_type_str == "float16") {
    maybe_group_scales_type = paddle::DataType::FLOAT16;
  }
  else if (maybe_group_scales_type_str == "bfloat16") {
    maybe_group_scales_type = paddle::DataType::BFLOAT16;
  }
  else if (maybe_group_scales_type_str == "float32") {
    maybe_group_scales_type = paddle::DataType::FLOAT32;
  }
  else if (maybe_group_scales_type_str == "") {
    maybe_group_scales_type = std::nullopt;
  }
  else {
    PADDLE_ENFORCE(false, "maybe_group_scales_type_str not supported!");
  }
  return machete::prepack_B_dispatch(
      {.B = B,
       .a_type = a_type,
       .b_type = b_type,
       .maybe_group_scales_type = maybe_group_scales_type});
}

std::vector<paddle::Tensor> MachetePrepackBKernel(
    paddle::Tensor const& B, std::string const& a_type_str, std::string const& b_type_str,
    std::string const& maybe_group_scales_type_str) {

  machete::ScalarTypeId b_type_id;
  paddle::DataType a_type, maybe_group_scales_type;

  if (b_type_str == "uint4b8") {
    b_type_id = machete::kU4B8.id();
  } else if (b_type_str == "uint8b128") {
    b_type_id = machete::kU8B128.id();
  } else {
    PADDLE_ENFORCE(false, "b_type_str not supported!");
  }

  if (a_type_str == "float16") {
    a_type = paddle::DataType::FLOAT16;
  }
  else if (a_type_str == "bfloat16") {
    a_type = paddle::DataType::BFLOAT16;
  }
  else {
    PADDLE_ENFORCE(false, "a_type_str not supported!");
  }
  auto Bt = paddle::experimental::transpose(B, {1, 0});
  paddle::Tensor B_prepacked = prepack_B(Bt, a_type, b_type_id, maybe_group_scales_type_str);
  return {B_prepacked};

}

std::vector<std::vector<int64_t>> MachetePrepackBKernelInferShape(
    std::vector<int64_t> const& B_shape, std::string const& a_type_str, std::string const& b_type_str,
    std::string const& maybe_group_scales_type_str) {
  return {{B_shape[1], B_shape[0]}};
}

std::vector<paddle::DataType> MachetePrepackBKernelInferDtype(
    paddle::DataType const& B_dtype, std::string const& a_type_str, std::string const& b_type_str,
    std::string const& maybe_group_scales_type_str) {
  return {B_dtype};
}

PD_BUILD_STATIC_OP(machete_prepack_B)
    .Inputs({"B"})
    .Outputs({"B_prepacked"})
    .Attrs({"a_type_str:std::string", "b_type_str:std::string", "maybe_group_scales_type_str:std::string"})
    .SetKernelFn(PD_KERNEL(MachetePrepackBKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(MachetePrepackBKernelInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MachetePrepackBKernelInferDtype));
