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

template <paddle::DataType T>
void MoeDispatchKernel(
    const paddle::Tensor &input,
    const paddle::Tensor &gating_output,
    const paddle::optional<paddle::Tensor> &gating_correction_bias,
    const paddle::optional<paddle::Tensor> &w4a8_in_scale,
    const int moe_topk,
    const bool group_moe,
    const bool topk_only_mode,
    const int num_rows,
    const int hidden_size,
    const int expert_num,
    paddle::Tensor *permute_input,
    paddle::Tensor *tokens_expert_prefix_sum,
    paddle::Tensor *permute_indices_per_token,
    paddle::Tensor *topk_weight,
    paddle::Tensor *topk_idx,
    paddle::Tensor *expert_idx_per_token,
    paddle::Tensor *dequant_scale) {
  using namespace phi;

  if (num_rows == 0) {
    return;
  }
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto stream = input.stream();
  auto place = input.place();

  if (group_moe) {
    // Check if expert_num is divisible by moe_topk, else throw an error
    PADDLE_ENFORCE_EQ(expert_num % moe_topk,
                      0,
                      common::errors::InvalidArgument(
                          "The number of experts (expert_num) "
                          "must be divisible by moe_topk. "
                          "Got expert_num = %d and moe_topk = %d.",
                          expert_num,
                          moe_topk));
  }

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

  int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();
  int *source_rows_ = reinterpret_cast<int *>(ws_ptr);
  int8_t *sorter_ws_ptr = reinterpret_cast<int8_t *>(ws_ptr + bytes);
  int *permuted_experts_ =
      reinterpret_cast<int *>(sorter_ws_ptr + sorter_ws_size_bytes);
  // If expected_ws_size > workspace_size ever occurs in sorter_.run (which
  // should be practically impossible), there is a contiguous, currently unused
  // region (permuted_experts_) right after sorter_ws_ptr. In practice, this
  // region is larger than what cub::DeviceRadixSort::SortPairs requires.
  // However, relying on this to “work” after canceling the assertion is unsafe:
  // it constitutes undefined behavior, and there is no guarantee it will remain
  // correct across inputs, CUDA/CUB versions, or architectures.
  int *permuted_rows_ = permuted_experts_ + num_moe_inputs;

  int *topk_idx_ptr = topk_idx->data<int>();

  float *softmax_max_prob = nullptr;
  if (group_moe) {
    paddle::Tensor softmax_max_prob_tensor =
        GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::FLOAT32, place);
    // (TODO: check fill success ?)
    paddle::experimental::fill(softmax_max_prob_tensor, 0.f);
    softmax_max_prob = softmax_max_prob_tensor.data<float>();
  }

  float *softmax_out_;

  const bool is_pow_2 =
      (expert_num != 0) && ((expert_num & (expert_num - 1)) == 0);

  paddle::Tensor softmax_buffer;

  if (!is_pow_2 || expert_num > 256 || group_moe || gating_correction_bias) {
    softmax_buffer = GetEmptyTensor(
        {num_rows * expert_num}, paddle::DataType::FLOAT32, place);
    softmax_out_ = softmax_buffer.data<float>();
  } else {
    softmax_out_ = nullptr;
  }

  topk_gating_softmax_kernelLauncher(
      gating_output.data<float>(),
      gating_correction_bias ? gating_correction_bias.get().data<float>()
                             : nullptr,
      topk_weight->data<float>(),
      softmax_out_,
      topk_idx_ptr,
      source_rows_,
      softmax_max_prob,
      num_rows,
      expert_num,
      moe_topk,
      group_moe,
      stream,
      topk_only_mode);

  sorter_.run(reinterpret_cast<void *>(sorter_ws_ptr),
              sorter_ws_size_bytes,
              topk_idx_ptr,
              expert_idx_per_token->data<int32_t>(),
              source_rows_,
              permuted_rows_,
              moe_topk * num_rows,
              false,
              stream);

  if (w4a8_in_scale) {
    if (permute_input->dtype() == paddle::DataType::INT8) {
      initialize_moe_routing_kernelLauncher(
          input.data<data_t>(),
          permute_input->data<int8_t>(),
          permuted_rows_,
          expert_idx_per_token->data<int32_t>(),
          w4a8_in_scale->data<float>(),
          permute_indices_per_token->data<int32_t>(),
          nullptr,
          num_rows,
          num_rows,
          hidden_size,
          moe_topk,
          stream);
    } else if (permute_input->dtype() == paddle::DataType::FLOAT8_E4M3FN) {
      initialize_moe_routing_kernelLauncher(
          input.data<data_t>(),
          permute_input->data<float8_e4m3fn>(),
          permuted_rows_,
          expert_idx_per_token->data<int32_t>(),
          w4a8_in_scale->data<float>(),
          permute_indices_per_token->data<int32_t>(),
          nullptr,
          num_rows,
          num_rows,
          hidden_size,
          moe_topk,
          stream);
    }
  } else {
    if (permute_input->dtype() == paddle::DataType::FLOAT8_E4M3FN) {
      initialize_moe_routing_kernelLauncher(
          input.data<data_t>(),
          permute_input->data<float8_e4m3fn>(),
          permuted_rows_,
          expert_idx_per_token->data<int32_t>(),
          nullptr,
          permute_indices_per_token->data<int32_t>(),
          dequant_scale->data<float>(),
          num_rows,
          num_rows,
          hidden_size,
          moe_topk,
          stream);
    } else {
      initialize_moe_routing_kernelLauncher(
          input.data<data_t>(),
          permute_input->data<data_t>(),
          permuted_rows_,
          expert_idx_per_token->data<int32_t>(),
          nullptr,
          permute_indices_per_token->data<int32_t>(),
          nullptr,
          num_rows,
          num_rows,
          hidden_size,
          moe_topk,
          stream);
    }
  }

  compute_total_rows_before_expert(expert_idx_per_token->data<int32_t>(),
                                   moe_topk * num_rows,
                                   expert_num,
                                   tokens_expert_prefix_sum->data<int64_t>(),
                                   stream);
}

std::vector<paddle::Tensor> MoeExpertDispatch(
    const paddle::Tensor &input,
    const paddle::Tensor &gating_output,
    const paddle::optional<paddle::Tensor> &gating_correction_bias,
    const paddle::optional<paddle::Tensor> &w4a8_in_scale,
    const int moe_topk,
    const bool group_moe,
    const std::string &moe_quant_type,
    const bool topk_only_mode) {
  const auto input_type = input.dtype();
  auto place = input.place();
  int token_rows = 0;
  auto input_dims = input.dims();
  auto gating_dims = gating_output.dims();
  const int expert_num = gating_dims[gating_dims.size() - 1];

  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = input.dims()[input_dims.size() - 1];

  auto permute_input_dtype = input_type;
  if (w4a8_in_scale) {
    if (moe_quant_type == "w4a8") {
      permute_input_dtype = paddle::DataType::INT8;
    } else if (moe_quant_type == "w4afp8") {
      permute_input_dtype = paddle::DataType::FLOAT8_E4M3FN;
    }
  } else {
    if (moe_quant_type == "w4afp8") {
      permute_input_dtype = paddle::DataType::FLOAT8_E4M3FN;
    }
  }

  auto permute_input = GetEmptyTensor(
      {moe_topk * num_rows, hidden_size}, permute_input_dtype, place);
  int dequant_scale_size = 1;
  if (moe_quant_type == "w4afp8" && !w4a8_in_scale) {
    dequant_scale_size = moe_topk * num_rows;
  }

  auto dequant_scale =
      GetEmptyTensor({dequant_scale_size}, paddle::DataType::FLOAT32, place);
  // correspond to the weighted coefficients of the results from each expert.
  auto topk_weight =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::FLOAT32, place);
  auto topk_idx =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::INT32, place);

  auto tokens_expert_prefix_sum =
      GetEmptyTensor({expert_num}, paddle::DataType::INT64, place);
  auto permute_indices_per_token =
      GetEmptyTensor({moe_topk, num_rows}, paddle::DataType::INT32, place);

  auto expert_idx_per_token =
      GetEmptyTensor({num_rows * moe_topk}, paddle::DataType::INT32, place);

  if (token_rows == 0) {
    return {permute_input,
            tokens_expert_prefix_sum,
            permute_indices_per_token,
            topk_weight,
            topk_idx,
            expert_idx_per_token,
            dequant_scale};
  }

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      MoeDispatchKernel<paddle::DataType::BFLOAT16>(input,
                                                    gating_output,
                                                    gating_correction_bias,
                                                    w4a8_in_scale,
                                                    moe_topk,
                                                    group_moe,
                                                    topk_only_mode,
                                                    num_rows,
                                                    hidden_size,
                                                    expert_num,
                                                    &permute_input,
                                                    &tokens_expert_prefix_sum,
                                                    &permute_indices_per_token,
                                                    &topk_weight,
                                                    &topk_idx,
                                                    &expert_idx_per_token,
                                                    &dequant_scale);
      break;
    case paddle::DataType::FLOAT16:
      MoeDispatchKernel<paddle::DataType::FLOAT16>(input,
                                                   gating_output,
                                                   gating_correction_bias,
                                                   w4a8_in_scale,
                                                   moe_topk,
                                                   group_moe,
                                                   topk_only_mode,
                                                   num_rows,
                                                   hidden_size,
                                                   expert_num,
                                                   &permute_input,
                                                   &tokens_expert_prefix_sum,
                                                   &permute_indices_per_token,
                                                   &topk_weight,
                                                   &topk_idx,
                                                   &expert_idx_per_token,
                                                   &dequant_scale);
      break;
    default:
      PD_THROW("Unsupported data type for MoeDispatchKernel");
  }
  return {permute_input,
          tokens_expert_prefix_sum,
          permute_indices_per_token,
          topk_weight,
          topk_idx,
          expert_idx_per_token,
          dequant_scale};
}

std::vector<std::vector<int64_t>> MoeExpertDispatchInferShape(
    const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &gating_output_shape,
    const paddle::optional<std::vector<int64_t>> &bias_shape,
    const int moe_topk) {
  int token_rows = -1;

  if (input_shape.size() == 3) {
    token_rows = input_shape[0] * input_shape[1];
  } else {
    token_rows = input_shape[0];
  }
  const int expert_num = gating_output_shape[gating_output_shape.size() - 1];
  const int num_rows = token_rows;
  const int hidden_size = input_shape[input_shape.size() - 1];
  const int permuted_rows = num_rows == -1 ? -1 : moe_topk * num_rows;

  return {{permuted_rows, hidden_size},
          {expert_num},
          {moe_topk, num_rows},
          {num_rows, moe_topk},
          {num_rows, moe_topk},
          {permuted_rows},
          {num_rows}};
}

std::vector<paddle::DataType> MoeExpertDispatchInferDtype(
    const paddle::DataType &input_dtype,
    const paddle::DataType &gating_output_dtype,
    const paddle::optional<paddle::DataType> &bias_type,
    const int moe_topk) {
  return {input_dtype,
          paddle::DataType::INT64,
          paddle::DataType::INT32,
          paddle::DataType::FLOAT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::FLOAT32};
}

/**
 * @brief Mixture of Experts (MoE) Expert Dispatch Operator
 *
 * This operator performs the following key functions:
 * 1. Computes top-k experts for each input token based on gating scores
 * 2. Permutes input tokens according to their selected experts for efficient
 * expert processing
 * 3. Computes prefix sums of tokens per expert for group_gemm optimization
 *
 * Inputs:
 *   - input: The input tensor to be routed to experts
 *            Shape: [total_tokens, hidden_size]
 *            dtype: bfloat16 or float16
 *   - gating_output: Gating network output scores for each token-expert pair
 *                   Shape: [total_tokens, expert_num]
 *                   dtype: must be float32
 *   - gating_correction_bias: Optional bias term for gating correction
 * (expert_num)
 *
 * Outputs:
 *   - permute_input: Permuted input tensor organized by expert
 *                   Shape: [moe_topk * total_tokens, hidden_size]
 *                   dtype: Same as input
 *   - tokens_expert_prefix_sum: Prefix sum array of token counts per expert for
 * group_gemm Shape: [expert_num] dtype: int64
 *   - permute_indices_per_token: Indices mapping for reconstructing original
 * order Shape: [moe_topk, total_tokens] dtype: int32
 *   - top_k_weight: Weight coefficients for combining expert outputs
 *                  Shape: [total_tokens, moe_topk]
 *                  dtype: float32
 *   - top_k_indices: Indices of selected top-k experts for each token
 *                   Shape: [total_tokens, moe_topk]
 *                   dtype: int32
 *
 * Attributes:
 *   - moe_topk: Number of experts to select for each token (k value in top-k
 * routing)
 *   - group_moe: Whether to perform group softmax within the operator
 *               (true: softmax is computed within groups of experts,
 *                false: standard softmax across all experts)
 *   - topk_only_mode: Operation mode selector
 *                    (true: only performs topk selection without softmax,
 *                     false: performs full softmax+topk computation)
 *
 * Note:
 * - The operator requires 2D input format [total_tokens, hidden_size]
 * - For optimal performance, expert_num should be a power of 2 when possible
 * - When group_moe is true, expert_num must be divisible by moe_topk
 */
PD_BUILD_STATIC_OP(moe_expert_dispatch)
    .Inputs({"input",
             "gating_output",
             paddle::Optional("gating_correction_bias"),
             paddle::Optional("w4a8_in_scale")})
    .Outputs({"permute_input",
              "tokens_expert_prefix_sum",
              "permute_indices_per_token",
              "topk_weight",
              "topk_idx",
              "expert_idx_per_token",
              "dequant_scale"})
    .Attrs({"moe_topk:int",
            "group_moe:bool",
            "moe_quant_type:std::string",
            "topk_only_mode:bool"})
    .SetKernelFn(PD_KERNEL(MoeExpertDispatch))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertDispatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertDispatchInferDtype));
