// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#include "w4afp8_gemm.h"
#include "helper.h"
#include "paddle/extension.h"
#include "w4afp8_gemm_template.h"
#include "weight_kernel.hpp"
#include "weight_scale_kernel.hpp"

template <typename T>
class NVTraits;

template <>
class NVTraits<__nv_fp8_e4m3> {
 public:
  typedef cutlass::float_e4m3_t data_t;
};

template <>
class NVTraits<__nv_bfloat16> {
 public:
  typedef cutlass::bfloat16_t data_t;
};

template <>
class NVTraits<half> {
 public:
  typedef cutlass::half_t data_t;
};

template <typename OutputType>
void DisPatchW4AFp8Gemm(const cutlass::float_e4m3_t* input,
                        const cutlass::float_e4m3_t* weight,
                        const int64_t* tokens,
                        const float* weight_scale,
                        const float* input_dequant_scale,
                        OutputType* out,
                        const int64_t token_padding_size,
                        const int64_t max_tokens,
                        const int Experts,
                        const int64_t M,
                        const int64_t K,
                        const int WeightScaleGroup,
                        cudaStream_t stream) {
  int kBlockN = 256;
  if constexpr (std::is_same_v<OutputType, cutlass::bfloat16_t>) {
    GEMM_SWITCH_BF16(M,
                     K,
                     Experts,
                     token_padding_size,
                     kBlockN,
                     WeightScaleGroup,
                     weight,
                     input,
                     out,
                     weight_scale,
                     input_dequant_scale,
                     tokens,
                     max_tokens,
                     stream)
  } else {
    PD_THROW("Only supported dtype in ['BFLOAT16'].");
  }
}

std::vector<paddle::Tensor> W4AFp8Gemm(
    const paddle::Tensor& input,
    const paddle::Tensor& weight,
    const paddle::Tensor&
        tokens,  // If tokenpadding=0, this tensor represents the prefix sum of
                 // tensors, otherwise it represents the number of tokens in
                 // each group
    const paddle::Tensor& weight_scale,
    const paddle::optional<paddle::Tensor>& input_dequant_scale,
    const int64_t token_padding_size,
    const int64_t max_tokens,
    const bool is_bfloat16) {
  const int Experts = weight.dims()[0];
  const int M = weight.dims()[1];
  const int K = weight.dims()[2] * 2;
  const int WeightScaleGroup =
      weight_scale.dims().size() == 2 ? K : weight_scale.dims()[3];

  if (input.dtype() != paddle::DataType::FLOAT8_E4M3FN) {
    PD_THROW("Only supported dtype in ['FLOAT8_E4M3FN'].");
  }

  if (token_padding_size == 0) {
    const int all_tokens = input.dims()[0];
    if (is_bfloat16) {
      paddle::Tensor out = paddle::empty(
          {all_tokens, M}, paddle::DataType::BFLOAT16, input.place());
      phi::dtype::bfloat16* out_data = out.data<phi::dtype::bfloat16>();
      DisPatchW4AFp8Gemm(
          reinterpret_cast<const cutlass::float_e4m3_t*>(
              input.data<phi::dtype::float8_e4m3fn>()),
          reinterpret_cast<const cutlass::float_e4m3_t*>(
              weight.data<uint8_t>()),
          tokens.data<int64_t>(),
          weight_scale.data<float>(),
          input_dequant_scale
              ? const_cast<float*>(input_dequant_scale.get().data<float>())
              : nullptr,
          reinterpret_cast<cutlass::bfloat16_t*>(out_data),
          token_padding_size,
          max_tokens,
          Experts,
          M,
          K,
          WeightScaleGroup,
          input.stream());
      return {out};
    } else {
      PD_THROW("Only supported dtype in ['BFLOAT16'].");
    }
  } else {
    if (is_bfloat16) {
      paddle::Tensor out = paddle::empty({Experts, token_padding_size, M},
                                         paddle::DataType::BFLOAT16,
                                         input.place());
      phi::dtype::bfloat16* out_data = out.data<phi::dtype::bfloat16>();
      DisPatchW4AFp8Gemm(
          reinterpret_cast<const cutlass::float_e4m3_t*>(
              input.data<phi::dtype::float8_e4m3fn>()),
          reinterpret_cast<const cutlass::float_e4m3_t*>(
              weight.data<uint8_t>()),
          tokens.data<int64_t>(),
          weight_scale.data<float>(),
          input_dequant_scale
              ? const_cast<float*>(input_dequant_scale.get().data<float>())
              : nullptr,
          reinterpret_cast<cutlass::bfloat16_t*>(out_data),
          token_padding_size,
          max_tokens,
          Experts,
          M,
          K,
          WeightScaleGroup,
          input.stream());
      return {out};
    } else {
      PD_THROW("Only supported dtype in ['BFLOAT16'].");
    }
  }
}

template <typename InputType, typename OutputType>
void DisPatchW4AFp8GemmWrapper(const InputType* input,
                               const InputType* weight,
                               const int64_t* total_rows_before_expert,
                               const float* input_dequant_scale,
                               const float* weight_scale,
                               OutputType* out,
                               const int64_t token_padding_size,
                               const int64_t max_tokens,
                               const int num_experts,
                               const int64_t M,
                               const int64_t K,
                               const int WeightScaleGroup,
                               cudaStream_t stream) {
  using InType = typename NVTraits<InputType>::data_t;
  using OutType = typename NVTraits<OutputType>::data_t;
  DisPatchW4AFp8Gemm(reinterpret_cast<const InType*>(input),
                     reinterpret_cast<const InType*>(weight),
                     total_rows_before_expert,
                     weight_scale,
                     input_dequant_scale,
                     reinterpret_cast<OutType*>(out),
                     token_padding_size,
                     max_tokens,
                     num_experts,
                     M,
                     K,
                     WeightScaleGroup,
                     stream);
}

PD_BUILD_STATIC_OP(w4afp8_gemm_scale_permute)
    .Inputs({"weight_scale"})
    .Outputs({"permute_scale"})
    .SetInplaceMap({{"weight_scale", "permute_scale"}})
    .SetKernelFn(PD_KERNEL(W4AFp8GemmScalePermute));

PD_BUILD_STATIC_OP(w4afp8_gemm)
    .Inputs({"input",
             "weight",
             "tokens",
             "weight_scale",
             paddle::Optional("input_dequant_scale")})
    .Outputs({"out"})
    .Attrs({"token_padding_size: int64_t",
            "max_tokens: int64_t",
            "is_bfloat16: bool"})
    .SetKernelFn(PD_KERNEL(W4AFp8Gemm));

PD_BUILD_STATIC_OP(w4afp8_gemm_weight_convert)
    .Inputs({"weight"})
    .Outputs({"converted_weight"})
    .SetKernelFn(PD_KERNEL(W4AFp8GemmWeightConvert));

template void DisPatchW4AFp8GemmWrapper<__nv_fp8_e4m3, __nv_bfloat16>(
    const __nv_fp8_e4m3* input,
    const __nv_fp8_e4m3* weight,
    const int64_t* tokens,
    const float* input_dequant_scale,
    const float* weight_scale,
    __nv_bfloat16* out,
    const int64_t token_padding_size,
    const int64_t max_tokens,
    const int num_experts,
    const int64_t M,
    const int64_t K,
    const int WeightScaleGroup,
    cudaStream_t stream);

template void DisPatchW4AFp8GemmWrapper<__nv_fp8_e4m3, half>(
    const __nv_fp8_e4m3* input,
    const __nv_fp8_e4m3* weight,
    const int64_t* tokens,
    const float* input_dequant_scale,
    const float* weight_scale,
    half* out,
    const int64_t token_padding_size,
    const int64_t max_tokens,
    const int num_experts,
    const int64_t M,
    const int64_t K,
    const int WeightScaleGroup,
    cudaStream_t stream);
