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
#include "moe/fast_hardamard_kernel.h"
#include "moe/fused_moe_helper.h"

template <paddle::DataType T>
void MoeFFNKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& ffn1_weight,
                  const paddle::Tensor& ffn2_weight,
                  const paddle::optional<paddle::Tensor>& ffn1_bias,
                  const paddle::optional<paddle::Tensor>& ffn1_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_in_scale,
                  const paddle::optional<paddle::Tensor>& expert_idx_per_token,
                  const std::string& quant_method,
                  paddle::Tensor ffn_out,
                  bool used_in_ep_low_latency) {
    using namespace phi;
    typedef PDTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto quant_mode = cutlass::epilogue::QuantMode::PerChannelQuant;

    auto ffn_out_data = ffn_out.data<data_t>();
    auto place = permute_input.place();
    auto stream = permute_input.stream();

    auto fp16_moe_gemm_runner = MoeGemmRunner<DataType_, cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kNone>>();
    auto int8_moe_gemm_runner = MoeGemmRunner<DataType_, cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kWeightOnlyInt8>>();
    auto int4_moe_gemm_runner = MoeGemmRunner<DataType_, cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kWeightOnlyInt4>>();
    auto w4a8_moe_gemm_runner = W4A8MoeGemmRunner<DataType_, int8_t, cutlass::uint4b_t>();

    assert(permute_input.dims().size() == 3 || permute_input.dims().size() == 2);

    const int num_experts = ffn1_weight.dims()[0];
    const int hidden_size = permute_input.dims()[permute_input.dims().size() - 1];

    assert(ffn1_weight.dims().size() == 3);
    int inter_dim = ffn1_weight.dims()[1] * ffn1_weight.dims()[2] / hidden_size;

    constexpr size_t workspace_size = 1 * 1024 * 1024 * 1024; // for nf4 stream-k
    Allocator* allocator = paddle::GetAllocator(place);
    Allocator::AllocationPtr workspace;
    if (quant_method == "weight_only_int4" || quant_method == "w4a8") {
        inter_dim = inter_dim * 2;
    }
    if (quant_method == "w4a8") {
        workspace = allocator->Allocate(
            SizeOf(paddle::DataType::INT8) * workspace_size);
    }

    const int64_t inter_size = inter_dim;


    int num_experts_ = num_experts;
    int num_max_tokens_per_expert;
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
        fc1_out_tensor = GetEmptyTensor(
            {expanded_active_expert_rows, inter_size}, T, place);
    }

    auto fc1_out = fc1_out_tensor.data<data_t>();

    using NvType = typename traits_::DataType;

    auto fc1_expert_biases =
        ffn1_bias
            ? const_cast<paddle::Tensor*>(ffn1_bias.get_ptr())->data<data_t>()
            : nullptr;

    // This is a trick.
    // expanded_active_expert_rows is not needed in variable group gemm.
    // but is needed in accommodating deepep low latency mode
    const int64_t total_rows_in_ll_else_minus1 = used_in_ep_low_latency ? expanded_active_expert_rows : -1;

    // When we tune the optimal configuration, we need the actual total_rows.
    const int64_t tune_total_rows = expanded_active_expert_rows;

    if (quant_method == "weight_only_int8") {
        typename cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kWeightOnlyInt8>::Arguments quant_args;
        int8_moe_gemm_runner.moe_gemm_bias_act(
            reinterpret_cast<const NvType*>(permute_input.data<data_t>()),
            reinterpret_cast<const uint8_t*>(ffn1_weight.data<int8_t>()),
            reinterpret_cast<const NvType*>(
                const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())
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
        typename cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kWeightOnlyInt4>::Arguments quant_args;
        int4_moe_gemm_runner.moe_gemm_bias_act(
            reinterpret_cast<const NvType*>(permute_input.data<data_t>()),
            reinterpret_cast<const cutlass::uint4b_t*>(
                ffn1_weight.data<int8_t>()),
            reinterpret_cast<const NvType*>(
                const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())
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
            reinterpret_cast<const int8_t *>(permute_input.data<int8_t>()),
            reinterpret_cast<const cutlass::uint4b_t *>(
                ffn1_weight.data<int8_t>()),
            quant_mode,
            reinterpret_cast<const NvType*>(
                const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())
                    ->data<data_t>()),
            nullptr, // ffn1_scale_dyquant
            nullptr, // nf4_look_up_table
            reinterpret_cast<NvType *>(fc1_out),
            const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
            total_rows_in_ll_else_minus1,
            tune_total_rows,
            inter_size,
            hidden_size,
            reinterpret_cast<char*>(workspace->ptr()),
            workspace_size,
            num_experts,
            stream);
    } else {
        typename cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kNone>::Arguments quant_args;
        fp16_moe_gemm_runner.moe_gemm_bias_act(
            reinterpret_cast<const NvType*>(permute_input.data<data_t>()),
            reinterpret_cast<const NvType*>(ffn1_weight.data<data_t>()),
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
        act_out_tensor = GroupSwigluWithMasked(fc1_out_tensor, tokens_expert_prefix_sum);
    } else {
        act_out_tensor = paddle::experimental::swiglu(fc1_out_tensor, nullptr);
    }
    auto act_out = act_out_tensor.data<data_t>();

    if (quant_method == "weight_only_int8") {
        typename cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kWeightOnlyInt8>::Arguments quant_args;
        int8_moe_gemm_runner.moe_gemm(
            reinterpret_cast<const NvType*>(act_out),
            reinterpret_cast<const uint8_t*>(ffn2_weight.data<int8_t>()),
            reinterpret_cast<const NvType*>(
                const_cast<paddle::Tensor*>(ffn2_scale.get_ptr())
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
        typename cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kWeightOnlyInt4>::Arguments quant_args;
        int4_moe_gemm_runner.moe_gemm(
            reinterpret_cast<const NvType*>(act_out),
            reinterpret_cast<const cutlass::uint4b_t*>(
                ffn2_weight.data<int8_t>()),
            reinterpret_cast<const NvType*>(
                const_cast<paddle::Tensor*>(ffn2_scale.get_ptr())
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
        data_t *ffn2_shift = nullptr;
        data_t *ffn2_smooth = nullptr;
        Allocator::AllocationPtr int8_act_out;
        int8_act_out = allocator->Allocate(
            SizeOf(paddle::DataType::INT8) * act_out_tensor.numel());
        MoeFastHardamardWrapper<data_t, int8_t>(
            act_out_tensor.data<data_t>(),
            expert_idx_per_token ? expert_idx_per_token.get().data<int64_t>() : nullptr,
            ffn2_shift, // ffn2_shift->data<T>(),
            ffn2_smooth, // ffn2_smooth->data<T>(),
            ffn2_in_scale ? const_cast<paddle::Tensor*>(ffn2_in_scale.get_ptr())->data<float>() : nullptr,
            1,
            127.0,
            -127.0,
            expanded_active_expert_rows,
            inter_size / 2,
            reinterpret_cast<int8_t *>(int8_act_out->ptr()),
            stream
        );
        w4a8_moe_gemm_runner.moe_gemm(
            reinterpret_cast<int8_t *>(int8_act_out->ptr()),
            reinterpret_cast<const cutlass::uint4b_t *>(
                ffn2_weight.data<int8_t>()),
            quant_mode,
            reinterpret_cast<const NvType*>(
                const_cast<paddle::Tensor*>(ffn2_scale.get_ptr())
                    ->data<data_t>()),
            nullptr, // ffn2_scale_dyquant
            nullptr, // reinterpret_cast<const int32_t*>(d_nf4_look_up_table), // nf4_look_up_table
            reinterpret_cast<NvType *>(ffn_out_data),
            const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
            total_rows_in_ll_else_minus1,
            tune_total_rows,
            hidden_size,
            inter_size / 2,
            reinterpret_cast<char*>(workspace->ptr()),
            workspace_size,
            num_experts,
            stream);
    } else {
        typename cutlass::WintQuantTraits<DataType_, cutlass::WintQuantMethod::kNone>::Arguments quant_args;
        fp16_moe_gemm_runner.moe_gemm(
            reinterpret_cast<const NvType*>(act_out),
            reinterpret_cast<const NvType*>(ffn2_weight.data<data_t>()),
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
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn2_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method, const bool used_in_ep_low_latency) {

    cudaCheckError();
    const auto t_type = quant_method == "w4a8" ? ffn1_scale.get().dtype() : permute_input.dtype();
    auto ffn_out = paddle::empty_like(permute_input, t_type);

    switch (t_type) {
        case paddle::DataType::BFLOAT16:
            MoeFFNKernel<paddle::DataType::BFLOAT16>(permute_input,
                                                     tokens_expert_prefix_sum,
                                                     ffn1_weight,
                                                     ffn2_weight,
                                                     ffn1_bias,
                                                     ffn1_scale,
                                                     ffn2_scale,
                                                     ffn2_in_scale,
                                                     expert_idx_per_token,
                                                     quant_method,
                                                     ffn_out, used_in_ep_low_latency);
            break;
        case paddle::DataType::FLOAT16:
            MoeFFNKernel<paddle::DataType::FLOAT16>(permute_input,
                                                    tokens_expert_prefix_sum,
                                                    ffn1_weight,
                                                    ffn2_weight,
                                                    ffn1_bias,
                                                    ffn1_scale,
                                                    ffn2_scale,
                                                    ffn2_in_scale,
                                                    expert_idx_per_token,
                                                    quant_method,
                                                    ffn_out, used_in_ep_low_latency);
            break;
        default:
            PD_THROW("Unsupported data type for MoeExpertFFN");
    }
    return ffn_out;
}

std::vector<paddle::Tensor> MoeExpertFFN(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn2_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method, const bool used_in_ep_low_latency) {
    return {MoeExpertFFNFunc(permute_input,
                             tokens_expert_prefix_sum,
                             ffn1_weight,
                             ffn2_weight,
                             ffn1_bias,
                             ffn1_scale,
                             ffn2_scale,
                             ffn2_in_scale,
                             expert_idx_per_token,
                             quant_method, used_in_ep_low_latency)};
}

std::vector<std::vector<int64_t>> MoeExpertFFNInferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& ffn1_weight_shape,
    const std::vector<int64_t>& ffn2_weight_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_in_scale_shape,
    const paddle::optional<std::vector<int64_t>>& expert_idx_per_token_shape,
    const std::string& quant_method,
    const bool used_in_ep_low_latency) {
    return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNInferDtype(
    const paddle::DataType &permute_input_dtype,
    const paddle::DataType &tokens_expert_prefix_sum_dtype,
    const paddle::DataType &ffn1_weight_dtype,
    const paddle::DataType &ffn2_weight_dtype,
    const paddle::optional<paddle::DataType> &ffn1_bias_dtype,
    const paddle::optional<paddle::DataType> &ffn1_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_in_scale_dtype,
    const std::string &quant_method, const bool used_in_ep_low_latency) {
  if (quant_method == "w4a8") {
    return {ffn1_scale_dtype.get()};
  } else {
    return {permute_input_dtype};
  }
}

/**
 * @brief Mixture of Experts (MoE) Feed-Forward Network Operator
 *
 * This operator performs the expert computation in MoE architecture, including:
 * 1. First linear transformation (FFN1) with optional quantization
 * 2. SwiGLU activation function
 * 3. Second linear transformation (FFN2) with optional quantization
 *
 * Supports multiple quantization methods including weight-only int4/int8 and w4a8 quantization.
 *
 * Inputs:
 *   - permute_input: Permuted input tensor organized by expert
 *                   Shape: [total_tokens * top_k, hidden_size]
 *                   dtype: bfloat16/float16 (or int8 for w4a8)
 *   - tokens_expert_prefix_sum: Prefix sum array of token counts per expert for group_gemm
 *                              Shape: [num_experts]
 *                              dtype: int64
 *   - ffn1_weight: First FFN layer weights
 *                 Shape: [num_experts, inter_size * 2, hidden_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - ffn2_weight: Second FFN layer weights
 *                 Shape: [num_experts, hidden_size, inter_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - ffn1_bias: Optional bias for first FFN layer
 *               Shape: [num_experts, inter_size * 2]
 *               dtype: Same as input
 *   - ffn1_scale: Quantization scales for first FFN layer
 *                Shape: [num_experts, inter_size * 2]
 *                dtype: Same as input
 *   - ffn2_scale: Quantization scales for second FFN layer
 *                Shape: [num_experts, hidden_size]
 *                dtype: Same as input
 *   - ffn2_in_scale: Optional input scales for second FFN layer (w4a8 only)
 *                   dtype: float32
 *   - expert_idx_per_token: Optional expert indices per token (w4a8 only)
 *                         Shape: [total_tokens]
 *                         dtype: int64
 *
 * Outputs:
 *   - output_tensor: Output tensor after MoE FFN computation
 *                   Shape: Same as permute_input
 *                   dtype: Same as input (or ffn1_scale dtype for w4a8)
 *
 * Attributes:
 *   - quant_method: Quantization method to use
 *                 Options: "none", "weight_only_int4", "weight_only_int8", "w4a8"
 *   - used_in_ep_low_latency: Whether running in low latency mode
 *                            Affects activation function implementation
 *
 * Note:
 * - w4a8 mode requires additional workspace memory allocation
 * - Low latency mode uses specialized grouped SwiGLU implementation
 */
PD_BUILD_STATIC_OP(moe_expert_ffn)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "ffn1_weight",
             "ffn2_weight",
             paddle::Optional("ffn1_bias"),
             paddle::Optional("ffn1_scale"),
             paddle::Optional("ffn2_scale"),
             paddle::Optional("ffn2_in_scale"),
             paddle::Optional("expert_idx_per_token")})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string", "used_in_ep_low_latency:bool"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFN))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNInferDtype));
