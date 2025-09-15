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


// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma once

#include "moe/fused_moe_helper.h"
#include "moe/fused_moe_op.h"
#pragma GCC diagnostic pop

#include "helper.h"
using namespace phi;

template <typename T>
void moe_redundant_topk_select_kernel(const T* input,
                            const T* bias,
                            T* output,
                            T* softmax,
                            const int* expert_id_to_ep_rank_array,
                            const int* expert_in_rank_num_list,
                            int* tokens_per_expert_stats_list,
                            int64_t* indices,
                            int64_t* indices_tmp,
                            int* source_row,
                            T* softmax_max_prob,
                            const int64_t num_rows,
                            const int64_t num_experts,
                            const int64_t k,
                            const int redundant_ep_rank_num_plus_one,
                            cudaStream_t stream,
                            const bool apply_norm_weight = false,
                            const bool enable_softmax_top_k_fused = false
                          ) {
  static constexpr int WARPS_PER_TB = 4;

  #define LAUNCH_TOPK_GATING_SOFTMAX_HELPER(N)                                   \
  case N: {                                                                    \
    topk_gating_softmax_launcher_helper<T, N, WARPS_PER_TB>(                   \
        input, bias, output, indices, source_row, num_rows, num_experts, k, stream); \
    break;                                                                     \
  }
  int64_t tem_num_experts = num_experts;
  if(bias != nullptr || apply_norm_weight)  tem_num_experts = 0;
  switch (tem_num_experts) {
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(2)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(4)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(8)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(16)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(32)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(64)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(128)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(256)

    default: {
      static constexpr int TPB = 256;
      const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
      if (!enable_softmax_top_k_fused) {
          moe_softmax<T, TPB><<<config_topk.block_per_grid, TPB, 0, stream>>>(
              input, softmax, num_experts, num_rows);
          if (apply_norm_weight) {
            moe_redundant_top_k_normed<T, TPB>
            <<<config_topk.block_per_grid, TPB, k * sizeof(T), stream>>>(softmax,
                                                             bias,
                                                             expert_id_to_ep_rank_array,
                                                             expert_in_rank_num_list,
                                                             tokens_per_expert_stats_list,
                                                             output,
                                                             indices,
                                                             indices_tmp,
                                                             source_row,
                                                             num_experts,
                                                             k,
                                                             num_rows,
                                                             redundant_ep_rank_num_plus_one);
          } else {
            moe_top_k<T, TPB>
                <<<config_topk.block_per_grid, TPB, 0, stream>>>(softmax,
                                                                  bias,
                                                                  output,
                                                                  indices,
                                                                  source_row,
                                                                  num_experts,
                                                                  k,
                                                                  num_rows);
          }
      }
      else {
          assert(k<=TPB);
          if (apply_norm_weight) {
            moe_softmax_top_k_fused<T, TPB, true>
                <<<config_topk.block_per_grid, TPB, k * sizeof(T), stream>>>(input,
                                                                 bias,
                                                                 output,
                                                                 indices,
                                                                 source_row,
                                                                 num_experts,
                                                                 k,
                                                                 num_rows);
          } else {
            moe_softmax_top_k_fused<T, TPB, false>
                <<<config_topk.block_per_grid, TPB, 0, stream>>>(input,
                                                                  bias,
                                                                  output,
                                                                  indices,
                                                                  source_row,
                                                                  num_experts,
                                                                  k,
                                                                  num_rows);
          }
      }

    }
  }
}

std::vector<paddle::Tensor> MoERedundantTopKSelectKernel(
    const paddle::Tensor& gating_logits,
    const paddle::Tensor& expert_id_to_ep_rank_array,
    const paddle::Tensor& expert_in_rank_num_list,
    paddle::Tensor& tokens_per_expert_stats_list,
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one) {
  auto stream = gating_logits.stream();
  auto place = gating_logits.place();
  int token_rows = 0;
  auto gating_dims = gating_logits.dims();
  const int expert_num = gating_dims[gating_dims.size() - 1];

  if (gating_dims.size() == 3) {
    token_rows = gating_dims[0] * gating_dims[1];
  } else {
    token_rows = gating_dims[0];
  }
  const int num_rows = token_rows;
  // correspond to the weighted coefficients of the results from each expert.
  auto topk_ids =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::INT64, place);
  auto topk_ids_tmp =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::INT64, place);
  auto topk_weights =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::FLOAT32, place);

  const int num_moe_inputs = AlignTo16(num_rows * moe_topk);
  const int bytes = num_moe_inputs * sizeof(int);

  CubKeyValueSorter sorter_;
  sorter_.update_num_experts(expert_num);

  const int sorter_ws_size_bytes =
      AlignTo16(sorter_.getWorkspaceSize(moe_topk * num_rows));
  const int sort_tmp_in_out_size = num_moe_inputs * 2 * sizeof(int);

  paddle::Tensor ws_ptr_tensor =
      GetEmptyTensor({bytes + sorter_ws_size_bytes + sort_tmp_in_out_size},
                     paddle::DataType::INT8,
                     place);

  int8_t* ws_ptr = ws_ptr_tensor.data<int8_t>();
  int* source_rows_ = reinterpret_cast<int*>(ws_ptr);

  int64_t* topk_ids_data = topk_ids.data<int64_t>();
  int64_t* topk_ids_tmp_data = topk_ids_tmp.data<int64_t>();

  float* softmax_max_prob = nullptr;
  float* softmax_out_;

  const bool is_pow_2 =
      (expert_num != 0) && ((expert_num & (expert_num - 1)) == 0);

  paddle::Tensor softmax_buffer;

  if (!is_pow_2 || expert_num > 256 || bias || apply_norm_weight) {
    softmax_buffer = GetEmptyTensor(
        {num_rows * expert_num}, paddle::DataType::FLOAT32, place);
    softmax_out_ = softmax_buffer.data<float>();
  } else {
    softmax_out_ = nullptr;
  }

  moe_redundant_topk_select_kernel<float>(gating_logits.data<float>(),
                                bias ? bias.get().data<float>() : nullptr,
                                topk_weights.data<float>(),
                                softmax_out_,
                                expert_id_to_ep_rank_array.data<int>(),
                                expert_in_rank_num_list.data<int>(),
                                tokens_per_expert_stats_list.data<int>(),
                                topk_ids_data,
                                topk_ids_tmp_data,
                                source_rows_,
                                softmax_max_prob,
                                num_rows,
                                expert_num,
                                moe_topk,
                                redundant_ep_rank_num_plus_one,
                                stream,
                                apply_norm_weight,
                                enable_softmax_top_k_fused);
  return {topk_ids, topk_weights};
}

std::vector<std::vector<int64_t>> MoERedundantTopKSelectKernelInferShape(
    const std::vector<int64_t>& gating_logits_shape,
    const std::vector<int64_t>& expert_id_to_ep_rank_array_shape,
    const std::vector<int64_t>& expert_in_rank_num_list_shape,
    const std::vector<int64_t>& tokens_per_expert_stats_list_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one) {
  int token_rows = -1;

  if (gating_logits_shape.size() == 3) {
    token_rows = gating_logits_shape[0] * gating_logits_shape[1];
  } else {
    token_rows = gating_logits_shape[0];
  }
  const int num_rows = token_rows;

  return {{num_rows, moe_topk},
          {num_rows, moe_topk}};
}

std::vector<paddle::DataType> MoERedundantTopKSelectKernelInferDtype(
    const paddle::DataType& gating_logits_dtype,
    const paddle::DataType& expert_id_to_ep_rank_array_dtype,
    const paddle::DataType& expert_in_rank_num_list_dtype,
    const paddle::DataType& tokens_per_expert_stats_list_dtype,
    const paddle::optional<paddle::DataType>& bias_type,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one) {
  return {paddle::DataType::INT64,
          paddle::DataType::FLOAT32};
}


PD_BUILD_STATIC_OP(moe_redundant_topk_select)
    .Inputs({"gating_logits", "expert_id_to_ep_rank_array", "expert_in_rank_num_list", "tokens_per_expert_stats_list", paddle::Optional("bias")})
    .Outputs({"topk_ids",
              "topk_weights",
              "tokens_per_expert_stats_list_out"})
    .Attrs({"moe_topk:int", "apply_norm_weight:bool", "enable_softmax_top_k_fused:bool", "redundant_ep_rank_num_plus_one:int"})
    .SetInplaceMap({{"tokens_per_expert_stats_list", "tokens_per_expert_stats_list_out"}})
    .SetKernelFn(PD_KERNEL(MoERedundantTopKSelectKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(MoERedundantTopKSelectKernelInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoERedundantTopKSelectKernelInferDtype));
