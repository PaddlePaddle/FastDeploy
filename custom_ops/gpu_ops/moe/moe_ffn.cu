// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "cutlass/numeric_conversion.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/epilogue/epilogue_quant_helper.h"
#include "cutlass_kernels/w4a8_moe/w4a8_moe_gemm_kernel.h"
#include "group_swiglu_with_masked.h"
#include "helper.h"
#include "moe/fused_moe_helper.h"
#include "moe/moe_fast_hardamard_kernel.h"
#include "swigluoai.h"
#include "w4afp8_gemm/w4afp8_gemm.h"

template <paddle::DataType T>
void MoeFFNKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& up_gate_proj_weight,
                  const paddle::Tensor& down_proj_weight,
                  const paddle::optional<paddle::Tensor>& up_proj_in_scale,
                  const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
                  const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
                  const paddle::optional<paddle::Tensor>& down_proj_scale,
                  const paddle::optional<paddle::Tensor>& down_proj_in_scale,
                  const paddle::optional<paddle::Tensor>& expert_idx_per_token,
                  const std::string& quant_method,
                  paddle::Tensor ffn_out,
                  bool used_in_ep_low_latency,
                  const int estimate_total_token_nums,
                  const int hadamard_block_size,
                  const std::string& activation) {
  using namespace phi;
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto quant_mode = cutlass::epilogue::QuantMode::PerChannelQuant;

  auto ffn_out_data = ffn_out.data<data_t>();
  auto place = permute_input.place();
  auto stream = permute_input.stream();

  auto fp16_moe_gemm_runner = MoeGemmRunner<
      DataType_,
      cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kNone>>();
  auto int8_moe_gemm_runner = MoeGemmRunner<
      DataType_,
      cutlass::WintQuantTraits<DataType_,
                               cutlass::WintQuantMethod::kWeightOnlyInt8>>();
  auto int4_moe_gemm_runner = MoeGemmRunner<
      DataType_,
      cutlass::WintQuantTraits<DataType_,
                               cutlass::WintQuantMethod::kWeightOnlyInt4>>();
  auto w4a8_moe_gemm_runner =
      W4A8MoeGemmRunner<DataType_, int8_t, cutlass::uint4b_t>();

  assert(permute_input.dims().size() == 3 || permute_input.dims().size() == 2);

  const int num_experts = up_gate_proj_weight.dims()[0];
  const int hidden_size = permute_input.dims()[permute_input.dims().size() - 1];

  assert(up_gate_proj_weight.dims().size() == 3);
  int inter_dim = up_gate_proj_weight.dims()[1] *
                  up_gate_proj_weight.dims()[2] / hidden_size;

  constexpr size_t workspace_size = 1 * 1024 * 1024 * 1024;  // for nf4 stream-k
  Allocator* allocator = paddle::GetAllocator(place);
  Allocator::AllocationPtr workspace;
  if (quant_method == "weight_only_int4" || quant_method == "w4a8" ||
      quant_method == "w4afp8") {
    inter_dim = inter_dim * 2;
  }
  if (quant_method == "w4a8" || quant_method == "w4afp8") {
    workspace =
        allocator->Allocate(SizeOf(paddle::DataType::INT8) * workspace_size);
  }

  const int64_t inter_size = inter_dim;

  typedef PDTraits<paddle::DataType::FLOAT8_E4M3FN> traits_fp8;
  typedef typename traits_fp8::DataType DataType_fp8;
  typedef typename traits_fp8::data_t data_t_fp8;

  int num_experts_ = num_experts;
  int num_max_tokens_per_expert = 256;
  int expanded_active_expert_rows;

  paddle::Tensor fc1_out_tensor;

  if (permute_input.dims().size() == 3) {
    num_experts_ = permute_input.dims()[0];
    assert(num_experts == num_experts_);

    num_max_tokens_per_expert = permute_input.dims()[1];
    expanded_active_expert_rows = num_experts_ * num_max_tokens_per_expert;
    fc1_out_tensor = GetEmptyTensor(
        {num_experts_, num_max_tokens_per_expert, inter_size}, T, place);
  } else {
    expanded_active_expert_rows = permute_input.dims()[0];
    fc1_out_tensor =
        GetEmptyTensor({expanded_active_expert_rows, inter_size}, T, place);
  }

  auto fc1_out = fc1_out_tensor.data<data_t>();

  using NvType = typename traits_::DataType;

  auto fc1_expert_biases =
      up_gate_proj_bias
          ? const_cast<paddle::Tensor*>(up_gate_proj_bias.get_ptr())
                ->data<data_t>()
          : nullptr;

  // This is a trick.
  // expanded_active_expert_rows is not needed in variable group gemm.
  // but is needed in accommodating deepep low latency mode
  const int64_t total_rows_in_ll_else_minus1 =
      used_in_ep_low_latency ? expanded_active_expert_rows : -1;

  // When we tune the optimal configuration, we need the actual total_rows.
  const int64_t tune_total_rows = expanded_active_expert_rows;

  if (quant_method == "weight_only_int8") {
    typename cutlass::WintQuantTraits<
        DataType_,
        cutlass::WintQuantMethod::kWeightOnlyInt8>::Arguments quant_args;
    int8_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permute_input.data<data_t>()),
        reinterpret_cast<const uint8_t*>(up_gate_proj_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(up_gate_proj_scale.get_ptr())
                ->data<data_t>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        tune_total_rows,
        inter_size,
        hidden_size,
        num_experts,
        quant_args,
        "none",
        stream);
  } else if (quant_method == "weight_only_int4") {
    typename cutlass::WintQuantTraits<
        DataType_,
        cutlass::WintQuantMethod::kWeightOnlyInt4>::Arguments quant_args;
    int4_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permute_input.data<data_t>()),
        reinterpret_cast<const cutlass::uint4b_t*>(
            up_gate_proj_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(up_gate_proj_scale.get_ptr())
                ->data<data_t>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        tune_total_rows,
        inter_size,
        hidden_size,
        num_experts,
        quant_args,
        "none",
        stream);
  } else if (quant_method == "w4a8") {
    w4a8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const int8_t*>(permute_input.data<int8_t>()),
        reinterpret_cast<const cutlass::uint4b_t*>(
            up_gate_proj_weight.data<int8_t>()),
        quant_mode,
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(up_gate_proj_scale.get_ptr())
                ->data<data_t>()),
        nullptr,  // up_gate_proj_scale_dyquant
        nullptr,  // nf4_look_up_table
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        used_in_ep_low_latency ? estimate_total_token_nums : tune_total_rows,
        inter_size,
        hidden_size,
        reinterpret_cast<char*>(workspace->ptr()),
        workspace_size,
        num_experts,
        stream);
  } else if (quant_method == "w4afp8") {
    typedef PDTraits<paddle::DataType::FLOAT8_E4M3FN> traits_fp8;
    typedef typename traits_fp8::DataType DataType_fp8;
    typedef typename traits_fp8::data_t data_t_fp8;
    paddle::Tensor weight_scale_tensor =
        *const_cast<paddle::Tensor*>(up_gate_proj_scale.get_ptr());
    const int weight_scale_group_size = weight_scale_tensor.dims().size() == 2
                                            ? hidden_size
                                            : weight_scale_tensor.dims()[3];
    const float* input_dequant_scale =
        up_proj_in_scale ? up_proj_in_scale.get().data<float>() : nullptr;
    DisPatchW4AFp8GemmWrapper(
        reinterpret_cast<const DataType_fp8*>(permute_input.data<data_t_fp8>()),
        reinterpret_cast<const DataType_fp8*>(
            up_gate_proj_weight.data<int8_t>()),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        input_dequant_scale,
        weight_scale_tensor.data<float>(),
        reinterpret_cast<NvType*>(fc1_out),
        used_in_ep_low_latency ? num_max_tokens_per_expert : 0,
        used_in_ep_low_latency ? num_max_tokens_per_expert
                               : permute_input.dims()[0],
        num_experts,
        inter_size,
        hidden_size,
        weight_scale_group_size,
        stream);
  } else {
    typename cutlass::WintQuantTraits<
        DataType_,
        cutlass::WintQuantMethod::kNone>::Arguments quant_args;
    fp16_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permute_input.data<data_t>()),
        reinterpret_cast<const NvType*>(up_gate_proj_weight.data<data_t>()),
        nullptr,
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        tune_total_rows,
        inter_size,
        hidden_size,
        num_experts,
        quant_args,
        "none",
        stream);
  }

  paddle::Tensor act_out_tensor;
  if (used_in_ep_low_latency) {
    act_out_tensor =
        GroupSwigluWithMasked(fc1_out_tensor, tokens_expert_prefix_sum);
  } else {
    if (activation == "swigluoai") {
      act_out_tensor = SwigluOAI(fc1_out_tensor, 1.702, 7.0, "interleave");
    } else {
      act_out_tensor = paddle::experimental::swiglu(fc1_out_tensor, nullptr);
    }
  }

  auto act_out = act_out_tensor.data<data_t>();
  if (quant_method == "weight_only_int8") {
    typename cutlass::WintQuantTraits<
        DataType_,
        cutlass::WintQuantMethod::kWeightOnlyInt8>::Arguments quant_args;
    int8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const uint8_t*>(down_proj_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(down_proj_scale.get_ptr())
                ->data<data_t>()),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        tune_total_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        quant_args,
        stream);

  } else if (quant_method == "weight_only_int4") {
    typename cutlass::WintQuantTraits<
        DataType_,
        cutlass::WintQuantMethod::kWeightOnlyInt4>::Arguments quant_args;
    int4_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const cutlass::uint4b_t*>(
            down_proj_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(down_proj_scale.get_ptr())
                ->data<data_t>()),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        tune_total_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        quant_args,
        stream);
  } else if (quant_method == "w4a8") {
    data_t* down_proj_shift = nullptr;
    data_t* down_proj_smooth = nullptr;
    Allocator::AllocationPtr int8_act_out;
    int8_act_out = allocator->Allocate(SizeOf(paddle::DataType::INT8) *
                                       act_out_tensor.numel());
    MoeFastHardamardWrapper<data_t, int8_t>(
        act_out_tensor.data<data_t>(),
        expert_idx_per_token ? expert_idx_per_token.get().data<int64_t>()
                             : nullptr,
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        down_proj_shift,   // down_proj_shift->data<T>(),
        down_proj_smooth,  // down_proj_smooth->data<T>(),
        down_proj_in_scale
            ? const_cast<paddle::Tensor*>(down_proj_in_scale.get_ptr())
                  ->data<float>()
            : nullptr,
        1,
        127.0,
        -127.0,
        expanded_active_expert_rows,
        inter_size / 2,
        num_max_tokens_per_expert,
        used_in_ep_low_latency,
        hadamard_block_size,
        reinterpret_cast<int8_t*>(int8_act_out->ptr()),
        stream);
    w4a8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<int8_t*>(int8_act_out->ptr()),
        reinterpret_cast<const cutlass::uint4b_t*>(
            down_proj_weight.data<int8_t>()),
        quant_mode,
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(down_proj_scale.get_ptr())
                ->data<data_t>()),
        nullptr,  // down_proj_scale_dyquant
        nullptr,  // reinterpret_cast<const int32_t*>(d_nf4_look_up_table), //
                  // nf4_look_up_table
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        used_in_ep_low_latency ? estimate_total_token_nums : tune_total_rows,
        hidden_size,
        inter_size / 2,
        reinterpret_cast<char*>(workspace->ptr()),
        workspace_size,
        num_experts,
        stream);
  } else if (quant_method == "w4afp8") {
    data_t* ffn2_shift = nullptr;
    data_t* ffn2_smooth = nullptr;
    float* input_dequant_scale = nullptr;
    Allocator::AllocationPtr fp8_act_out;
    fp8_act_out = allocator->Allocate(SizeOf(paddle::DataType::INT8) *
                                      act_out_tensor.numel());

    if (down_proj_in_scale) {
      MoeFastHardamardWrapper<data_t, data_t_fp8>(
          act_out_tensor.data<data_t>(),
          expert_idx_per_token ? expert_idx_per_token.get().data<int64_t>()
                               : nullptr,
          const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
          ffn2_shift,
          ffn2_smooth,
          down_proj_in_scale
              ? const_cast<paddle::Tensor*>(down_proj_in_scale.get_ptr())
                    ->data<float>()
              : nullptr,
          1,
          448.0f,
          -448.0f,
          expanded_active_expert_rows,
          inter_size / 2,
          num_max_tokens_per_expert,
          used_in_ep_low_latency,
          hadamard_block_size,
          reinterpret_cast<data_t_fp8*>(fp8_act_out->ptr()),
          stream);
    } else {
      Allocator::AllocationPtr ffn2_input_dequant_scale;
      ffn2_input_dequant_scale =
          allocator->Allocate(sizeof(float) * expanded_active_expert_rows);
      input_dequant_scale =
          reinterpret_cast<float*>(ffn2_input_dequant_scale->ptr());
      MoeFastHardamardWrapper<data_t, data_t>(
          act_out_tensor.data<data_t>(),
          expert_idx_per_token ? expert_idx_per_token.get().data<int64_t>()
                               : nullptr,
          const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
          ffn2_shift,   // ffn2_shift->data<T>(),
          ffn2_smooth,  // ffn2_smooth->data<T>(),
          nullptr,
          1,
          448.0f,
          -448.0f,
          expanded_active_expert_rows,
          inter_size / 2,
          num_max_tokens_per_expert,
          used_in_ep_low_latency,
          hadamard_block_size,
          act_out_tensor.data<data_t>(),
          stream);

      quantize_moe_input<data_t, data_t_fp8>(
          act_out_tensor.data<data_t>(),
          expert_idx_per_token ? expert_idx_per_token.get().data<int64_t>()
                               : nullptr,
          expanded_active_expert_rows,
          inter_size / 2,
          input_dequant_scale,
          const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
          num_max_tokens_per_expert,
          used_in_ep_low_latency,
          reinterpret_cast<data_t_fp8*>(fp8_act_out->ptr()),
          stream);
    }

    paddle::Tensor weight_scale_tensor =
        *const_cast<paddle::Tensor*>(down_proj_scale.get_ptr());
    const int weight_scale_group_size = weight_scale_tensor.dims().size() == 2
                                            ? inter_size / 2
                                            : weight_scale_tensor.dims()[3];

    DisPatchW4AFp8GemmWrapper(
        reinterpret_cast<const DataType_fp8*>(fp8_act_out->ptr()),
        reinterpret_cast<const DataType_fp8*>(down_proj_weight.data<int8_t>()),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        input_dequant_scale,
        weight_scale_tensor.data<float>(),
        reinterpret_cast<NvType*>(ffn_out_data),
        used_in_ep_low_latency ? num_max_tokens_per_expert : 0,
        used_in_ep_low_latency ? num_max_tokens_per_expert
                               : act_out_tensor.dims()[0],
        num_experts,
        hidden_size,
        inter_size / 2,
        weight_scale_group_size,
        stream);
  } else {
    typename cutlass::WintQuantTraits<
        DataType_,
        cutlass::WintQuantMethod::kNone>::Arguments quant_args;
    fp16_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const NvType*>(down_proj_weight.data<data_t>()),
        nullptr,
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        tune_total_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        quant_args,
        stream);
  }
}

paddle::Tensor MoeExpertFFNFunc(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_proj_in_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method,
    const bool used_in_ep_low_latency,
    const int estimate_total_token_nums,
    const int hadamard_block_size,
    const std::string& activation) {
  const auto t_type = (quant_method == "w4a8")
                          ? up_gate_proj_scale.get().dtype()
                      : (quant_method == "w4afp8") ? paddle::DataType::BFLOAT16
                                                   : permute_input.dtype();
  auto ffn_out = paddle::empty_like(permute_input, t_type);
  if (permute_input.numel() == 0) {
    return ffn_out;
  }
  switch (t_type) {
    case paddle::DataType::BFLOAT16:
      MoeFFNKernel<paddle::DataType::BFLOAT16>(permute_input,
                                               tokens_expert_prefix_sum,
                                               up_gate_proj_weight,
                                               down_proj_weight,
                                               up_proj_in_scale,
                                               up_gate_proj_bias,
                                               up_gate_proj_scale,
                                               down_proj_scale,
                                               down_proj_in_scale,
                                               expert_idx_per_token,
                                               quant_method,
                                               ffn_out,
                                               used_in_ep_low_latency,
                                               estimate_total_token_nums,
                                               hadamard_block_size,
                                               activation);
      break;
    case paddle::DataType::FLOAT16:
      MoeFFNKernel<paddle::DataType::FLOAT16>(permute_input,
                                              tokens_expert_prefix_sum,
                                              up_gate_proj_weight,
                                              down_proj_weight,
                                              up_proj_in_scale,
                                              up_gate_proj_bias,
                                              up_gate_proj_scale,
                                              down_proj_scale,
                                              down_proj_in_scale,
                                              expert_idx_per_token,
                                              quant_method,
                                              ffn_out,
                                              used_in_ep_low_latency,
                                              estimate_total_token_nums,
                                              hadamard_block_size,
                                              activation);
      break;
    default:
      PD_THROW("Unsupported data type for MoeExpertFFN");
  }
  return ffn_out;
}

std::vector<paddle::Tensor> MoeExpertFFN(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_proj_in_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method,
    const bool used_in_ep_low_latency,
    const int estimate_total_token_nums,
    const int hadamard_block_size,
    const std::string& activation) {
  return {MoeExpertFFNFunc(permute_input,
                           tokens_expert_prefix_sum,
                           up_gate_proj_weight,
                           down_proj_weight,
                           up_proj_in_scale,
                           up_gate_proj_bias,
                           up_gate_proj_scale,
                           down_proj_scale,
                           down_proj_in_scale,
                           expert_idx_per_token,
                           quant_method,
                           used_in_ep_low_latency,
                           estimate_total_token_nums,
                           hadamard_block_size,
                           activation)};
}

std::vector<std::vector<int64_t>> MoeExpertFFNInferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& up_gate_proj_weight_shape,
    const std::vector<int64_t>& down_proj_weight_shape,
    const paddle::optional<std::vector<int64_t>>& up_proj_in_scale_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_in_scale_shape,
    const paddle::optional<std::vector<int64_t>>& expert_idx_per_token_shape,
    const std::string& quant_method,
    const bool used_in_ep_low_latency,
    const int estimate_total_token_nums,
    const int hadamard_block_size,
    const std::string& activation) {
  return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNInferDtype(
    const paddle::DataType& permute_input_dtype,
    const paddle::DataType& tokens_expert_prefix_sum_dtype,
    const paddle::DataType& up_gate_proj_weight_dtype,
    const paddle::DataType& down_proj_weight_dtype,
    const paddle::optional<paddle::DataType>& up_proj_in_scale_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_bias_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_in_scale_dtype,
    const std::string& quant_method,
    const bool used_in_ep_low_latency,
    const int estimate_total_token_nums,
    const int hadamard_block_size,
    const std::string& activation) {
  if (quant_method == "w4a8" || quant_method == "w4afp8") {
    return {up_gate_proj_scale_dtype.get()};
  } else {
    return {permute_input_dtype};
  }
}

/**
 * @brief Mixture of Experts (MoE) Feed-Forward Network Operator
 *
 * This operator performs the expert computation in MoE architecture, including:
 * 1. First linear transformation (up_gate_proj) with optional quantization
 * 2. SwiGLU activation function
 * 3. Second linear transformation (down_proj) with optional quantization
 *
 * Supports multiple quantization methods including weight-only int4/int8 and
 * w4a8 quantization.
 *
 * Inputs:
 *   - permute_input: Permuted input tensor organized by expert
 *                   Shape: [total_tokens * top_k, hidden_size]
 *                   dtype: bfloat16/float16 (or int8 for w4a8)
 *   - tokens_expert_prefix_sum: Prefix sum array of token counts per expert for
 * group_gemm Shape: [num_experts] dtype: int64
 *   - up_gate_proj_weight: First FFN layer weights
 *                 Shape: [num_experts, inter_size * 2, hidden_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - down_proj_weight: Second FFN layer weights
 *                 Shape: [num_experts, hidden_size, inter_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - up_gate_proj_bias: Optional bias for first FFN layer
 *               Shape: [num_experts, inter_size * 2]
 *               dtype: Same as input
 *   - up_gate_proj_scale: Quantization scales for first FFN layer
 *                Shape: [num_experts, inter_size * 2]
 *                dtype: Same as input
 *   - down_proj_scale: Quantization scales for second FFN layer
 *                Shape: [num_experts, hidden_size]
 *                dtype: Same as input
 *   - down_proj_in_scale: Optional input scales for second FFN layer (w4a8
 * only) dtype: float32
 *   - expert_idx_per_token: Optional expert indices per token (w4a8 only)
 *                         Shape: [total_tokens]
 *                         dtype: int64
 *
 * Outputs:
 *   - output_tensor: Output tensor after MoE FFN computation
 *                   Shape: Same as permute_input
 *                   dtype: Same as input (or up_gate_proj_scale dtype for w4a8)
 *
 * Attributes:
 *   - quant_method: Quantization method to use
 *                 Options: "none", "weight_only_int4", "weight_only_int8",
 * "w4a8"
 *   - used_in_ep_low_latency: Whether running in low latency mode
 *                            Affects activation function implementation
 *   - estimate_total_token_nums: estimate total token numbers
 *   - hadamard_block_size: hadamard block size for w4a8/w4afp8 quantization
 *
 * Note:
 * - w4a8 mode requires additional workspace memory allocation
 * - Low latency mode uses specialized grouped SwiGLU implementation
 */
PD_BUILD_STATIC_OP(moe_expert_ffn)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "up_gate_proj_weight",
             "down_proj_weight",
             paddle::Optional("up_proj_in_scale"),
             paddle::Optional("up_gate_proj_bias"),
             paddle::Optional("up_gate_proj_scale"),
             paddle::Optional("down_proj_scale"),
             paddle::Optional("down_proj_in_scale"),
             paddle::Optional("expert_idx_per_token")})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string",
            "used_in_ep_low_latency:bool",
            "estimate_total_token_nums:int",
            "hadamard_block_size:int",
            "activation:std::string"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFN))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNInferDtype));
