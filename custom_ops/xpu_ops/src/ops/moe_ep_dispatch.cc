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

#include <infer_ops.h>
#include <infer_ops_eb.h>
#include <xft_api.h>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename TX, typename TY>
std::vector<paddle::Tensor> EPMoeExpertDispatchKernel(
    const paddle::Tensor& input,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::optional<paddle::Tensor>& input_scales,
    const std::vector<int>& token_nums_per_expert,
    const int64_t token_nums_this_rank) {
  using XPU_TX = typename XPUTypeTrait<TX>::Type;
  using XPU_TY = typename XPUTypeTrait<TY>::Type;
  phi::XPUPlace xpu_place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(xpu_place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const auto input_type = input.dtype();
  auto m = input.dims()[0];
  auto n = input.dims()[1];
  const int64_t expert_num = token_nums_per_expert.size();
  const int topk = topk_ids.dims()[1];
  auto place = input.place();

  auto block_num = xpu_ctx->x_context()->ncluster();
  paddle::Tensor permute_input;
  auto permute_indices_per_token =
      paddle::empty({m, topk}, paddle::DataType::INT32, place);
  auto expert_m = paddle::empty({expert_num}, paddle::DataType::INT32, place);
  auto recv_num_tokens_per_expert_list_cumsum =
      paddle::empty({expert_num + 1}, paddle::DataType::INT32, place);
  auto expand_input_scales =
      paddle::empty({token_nums_this_rank}, paddle::DataType::FLOAT32, place);
  const int64_t ep_size = 1;
  const int64_t ep_rank = 0;

  if (std::is_same<TY, int8_t>::value) {
    permute_input =
        paddle::empty({token_nums_this_rank, n}, paddle::DataType::INT8, place);
    if (token_nums_this_rank > 0) {
      auto ret = infer_ops::moe_ffn_pre_sorted_quant_pe<XPU_TX, int>(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPU_TX*>(input.data<TX>()),
          topk_ids.data<int>(),
          input_scales.get_ptr()->data<float>(),
          nullptr,
          reinterpret_cast<int8_t*>(permute_input.data<int8_t>()),
          const_cast<int*>(permute_indices_per_token.data<int>()),
          const_cast<int*>(expert_m.data<int>()),
          const_cast<int*>(recv_num_tokens_per_expert_list_cumsum.data<int>()),
          expand_input_scales.data<float>(),
          m,
          n,
          expert_num,
          topk,
          block_num,
          token_nums_this_rank);
      PD_CHECK(ret == 0, "moe_ep_ffn_pre_sorted failed");
    }
  } else {
    permute_input = paddle::empty({token_nums_this_rank, n}, input_type, place);
    if (token_nums_this_rank > 0) {
      auto ret = infer_ops::moe_ep_ffn_pre_sorted<XPU_TX, int>(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPU_TX*>(input.data<TX>()),
          topk_ids.data<int>(),
          nullptr,
          reinterpret_cast<XPU_TX*>(permute_input.data<TX>()),
          const_cast<int*>(permute_indices_per_token.data<int>()),
          const_cast<int*>(expert_m.data<int>()),
          const_cast<int*>(recv_num_tokens_per_expert_list_cumsum.data<int>()),
          m,
          n,
          expert_num,
          topk,
          block_num,
          ep_size,
          ep_rank,
          token_nums_this_rank);
      PD_CHECK(ret == 0, "moe_ep_ffn_pre_sorted failed");
    }
  }
  return {permute_input,
          permute_indices_per_token,
          recv_num_tokens_per_expert_list_cumsum,
          topk_weights,
          expand_input_scales};
}

std::vector<paddle::Tensor> EPMoeExpertDispatch(
    const paddle::Tensor& input,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::optional<paddle::Tensor>& input_scales,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string quant_method) {
#define APPLY_KERNEL(TX, TY)                                      \
  return EPMoeExpertDispatchKernel<TX, TY>(input,                 \
                                           topk_ids,              \
                                           topk_weights,          \
                                           input_scales,          \
                                           token_nums_per_expert, \
                                           token_nums_this_rank);

  const auto input_dtype = input.dtype();
  if (input_dtype == paddle::DataType::FLOAT16 && quant_method == "w4a8") {
    APPLY_KERNEL(paddle::float16, int8_t);
  } else if (input_dtype == paddle::DataType::FLOAT16 &&
             quant_method != "w4a8") {
    APPLY_KERNEL(paddle::float16, paddle::float16);
  } else if (input_dtype == paddle::DataType::BFLOAT16 &&
             quant_method == "w4a8") {
    APPLY_KERNEL(paddle::bfloat16, int8_t);
  } else if (input_dtype == paddle::DataType::BFLOAT16 &&
             quant_method != "w4a8") {
    APPLY_KERNEL(paddle::bfloat16, paddle::bfloat16);
  } else {
    PD_THROW("EPMoeExpertDispatch not support input_dtype=",
             static_cast<int>(input_dtype),
             "quant_method=",
             quant_method);
    return {};
  }

#undef APPLY_KERNEL
}

std::vector<std::vector<int64_t>> EPMoeExpertDispatchInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& topk_ids_shape,
    const std::vector<int64_t>& topk_weights_shape,
    const paddle::optional<std::vector<int64_t>>& input_scales_shape,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string quant_method) {
  const int m = input_shape[0];
  const int hidden_size = input_shape[input_shape.size() - 1];
  const int topk = topk_ids_shape[topk_ids_shape.size() - 1];
  const int expert_num = token_nums_per_expert.size();
  return {{token_nums_this_rank, hidden_size},
          {expert_num, m},
          {expert_num},
          {token_nums_this_rank},
          {token_nums_this_rank}};
}

std::vector<paddle::DataType> EPMoeExpertDispatchInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& topk_ids_dtype,
    const paddle::DataType& topk_weights_dtype,
    const paddle::optional<paddle::DataType>& input_scales_dtype,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string quant_method) {
  auto output_dtype = input_dtype;
  if (quant_method == "w4a8") {
    output_dtype = paddle::DataType::INT8;
  }
  return {
      output_dtype,
      paddle::DataType::INT32,
      paddle::DataType::INT32,
      topk_weights_dtype,
      paddle::DataType::FLOAT32,
  };
}

PD_BUILD_STATIC_OP(ep_moe_expert_dispatch)
    .Inputs(
        {"input", "topk_ids", "topk_weights", paddle::Optional("input_scales")})
    .Outputs({"permute_input",
              "permute_indices_per_token",
              "token_nums_per_expert_cumsum",
              "dst_weights",
              "expand_input_scales"})
    .Attrs({"token_nums_per_expert: std::vector<int>",
            "token_nums_this_rank: int",
            "quant_method: std::string"})
    .SetKernelFn(PD_KERNEL(EPMoeExpertDispatch))
    .SetInferShapeFn(PD_INFER_SHAPE(EPMoeExpertDispatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(EPMoeExpertDispatchInferDtype));
