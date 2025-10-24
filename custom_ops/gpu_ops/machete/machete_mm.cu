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

template <typename T>
std::optional<T> ConvertToStdOptional(const paddle::optional<T>& paddle_opt) {
    return paddle_opt ? std::optional<T>(paddle_opt.get()) : std::nullopt;
}

paddle::Tensor mm(paddle::Tensor const& A, paddle::Tensor const& B,
                 int64_t b_type_id,
                 std::optional<paddle::DataType> const& maybe_out_type,
                 std::optional<paddle::Tensor> const& maybe_group_scales,
                 std::optional<paddle::Tensor> const& maybe_group_zeros,
                 int64_t maybe_group_size,
                 std::optional<paddle::Tensor> const& maybe_channel_scales,
                 std::optional<paddle::Tensor> const& maybe_token_scales,
                 std::string maybe_schedule) {
  machete::ScalarType const b_type = machete::ScalarType::from_id(b_type_id);
  std::optional<int64_t> maybe_group_size_opt = std::optional<int64_t>(maybe_group_size);
  std::optional<std::string> maybe_schedule_opt;
  if (maybe_schedule == "") {
    maybe_schedule_opt = std::nullopt;
  } else {
    maybe_schedule_opt = std::optional<std::string>(maybe_schedule);
  }
  return machete::mm_dispatch({.A = A,
                      .B = B,
                      .b_type = b_type,
                      .maybe_out_type = maybe_out_type,
                      .maybe_group_scales = maybe_group_scales,
                      .maybe_group_zeros = maybe_group_zeros,
                      .maybe_group_size = maybe_group_size_opt,
                      .maybe_channel_scales = maybe_channel_scales,
                      .maybe_token_scales = maybe_token_scales,
                      .maybe_schedule = maybe_schedule_opt});
}

std::vector<paddle::Tensor> MacheteMMKernel(
    paddle::Tensor const& A, paddle::Tensor const& B,
    paddle::optional<paddle::Tensor> const& maybe_group_scales,
    paddle::optional<paddle::Tensor> const& maybe_group_zeros,
    paddle::optional<paddle::Tensor> const& maybe_channel_scales,
    paddle::optional<paddle::Tensor> const& maybe_token_scales,
    std::string const& b_type_str,
    std::string const& maybe_out_type_str,
    int64_t const& maybe_group_size,
    std::string const& maybe_schedule
  ) {

  machete::ScalarTypeId b_type_id;
  paddle::DataType maybe_out_type;
  if (b_type_str == "uint4b8") {
    b_type_id = machete::kU4B8.id();
  } else if (b_type_str == "uint8b128") {
    b_type_id = machete::kU8B128.id();
  } else {
    PADDLE_ENFORCE(false, "b_type_str not supported!");
  }
  if (maybe_out_type_str == "float16") {
    maybe_out_type = paddle::DataType::FLOAT16;
  } else if (maybe_out_type_str == "bfloat16") {
    maybe_out_type = paddle::DataType::BFLOAT16;
  } else {
    maybe_out_type = A.dtype();
  }
  auto out = mm(A, B, b_type_id, maybe_out_type,
                ConvertToStdOptional<paddle::Tensor>(maybe_group_scales),
                ConvertToStdOptional<paddle::Tensor>(maybe_group_zeros),
                maybe_group_size,
                ConvertToStdOptional<paddle::Tensor>(maybe_channel_scales),
                ConvertToStdOptional<paddle::Tensor>(maybe_token_scales),
                maybe_schedule);
  return {out};
}

std::vector<std::vector<int64_t>> MacheteMMKernelInferShape(
    std::vector<int64_t> const& A_shape,
    std::vector<int64_t> const& B_shape,
    paddle::optional<std::vector<int64_t>> const& maybe_group_scales_shape,
    paddle::optional<std::vector<int64_t>> const& maybe_group_zeros_shape,
    paddle::optional<std::vector<int64_t>> const& maybe_channel_scales_shape,
    paddle::optional<std::vector<int64_t>> const& maybe_token_scales_shape,
    std::string const& b_type_str,
    std::string const& maybe_out_type_str,
    int64_t const& maybe_group_size,
    std::string const& maybe_schedule) {
  return {{A_shape[0], B_shape[1]}};
}

std::vector<paddle::DataType> MacheteMMKernelInferDtype(
    paddle::DataType const& A_dtype,
    paddle::DataType const& B_dtype,
    paddle::optional<paddle::DataType> const& maybe_group_scales_dtype,
    paddle::optional<paddle::DataType> const& maybe_group_zeros_dtype,
    paddle::optional<paddle::DataType> const& maybe_channel_scales_dtype,
    paddle::optional<paddle::DataType> const& maybe_token_scales_dtype,
    std::string const& b_type_str,
    std::string const& maybe_out_type_str,
    int64_t const& maybe_group_size,
    std::string const& maybe_schedule) {

  paddle::DataType maybe_out_type;
  if (maybe_out_type_str == "float16") {
    maybe_out_type = paddle::DataType::FLOAT16;
  } else if (maybe_out_type_str == "bfloat16") {
    maybe_out_type = paddle::DataType::BFLOAT16;
  } else {
    maybe_out_type = A_dtype;
  }
  return {maybe_out_type};
}

PD_BUILD_STATIC_OP(machete_mm)
    .Inputs({"A", "B",
             paddle::Optional("maybe_group_scales"),
             paddle::Optional("maybe_group_zeros"),
             paddle::Optional("maybe_channel_scales"),
             paddle::Optional("maybe_token_scales")})
    .Outputs({"out"})
    .Attrs({"b_type_str:std::string", "maybe_out_type_str:std::string", "maybe_group_size:int64_t", "maybe_schedule:std::string"})
    .SetKernelFn(PD_KERNEL(MacheteMMKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(MacheteMMKernelInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MacheteMMKernelInferDtype));
