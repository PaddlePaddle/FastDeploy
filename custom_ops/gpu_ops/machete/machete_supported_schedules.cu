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

std::vector<std::string> supported_schedules(
    paddle::DataType a_type, int64_t b_type_id,
    std::optional<paddle::DataType> maybe_group_scales_type,
    std::optional<paddle::DataType> maybe_group_zeros_type,
    std::optional<paddle::DataType> maybe_channel_scales_type,
    std::optional<paddle::DataType> maybe_token_scales_type,
    std::optional<paddle::DataType> maybe_out_type) {
  machete::ScalarType const b_type = machete::ScalarType::from_id(b_type_id);
  auto schedules = machete::supported_schedules_dispatch({
      .a_type = a_type,
      .b_type = b_type,
      .maybe_group_scales_type = maybe_group_scales_type,
      .maybe_group_zeros_type = maybe_group_zeros_type,
      .maybe_channel_scales_type = maybe_channel_scales_type,
      .maybe_token_scales_type = maybe_token_scales_type,
      .maybe_out_type = maybe_out_type
  });
  return schedules;
}

std::vector<std::string> MacheteSupportedSchedules(
    std::string const& a_type_str, std::string const& b_type_str) {
  machete::ScalarTypeId b_type_id;
  paddle::DataType a_type;
  if (b_type_str == "uint4b8") {
    b_type_id = machete::kU4B8.id();
  } else {
    PADDLE_ENFORCE(false, "b_type_str not supported!");
  }
  if (a_type_str == "bfloat16") {
    a_type = paddle::DataType::BFLOAT16;
  } else if (a_type_str == "float16") {
    a_type = paddle::DataType::FLOAT16;
  } else {
    PADDLE_ENFORCE(false, "a_type_str not supported!");
  }
  std::optional<paddle::DataType> maybe_group_scales_type = std::optional<paddle::DataType>(a_type);
  std::optional<paddle::DataType> maybe_out_type = std::optional<paddle::DataType>(a_type);
  std::optional<paddle::DataType> maybe_group_zeros_type = std::nullopt;
  std::optional<paddle::DataType> maybe_channel_scales_type = std::nullopt;
  std::optional<paddle::DataType> maybe_token_scales_type = std::nullopt;

  auto schedules =  supported_schedules(a_type, b_type_id,
        maybe_group_scales_type,
        maybe_group_zeros_type,
        maybe_channel_scales_type,
        maybe_token_scales_type,
        maybe_out_type);
  return schedules;
}
