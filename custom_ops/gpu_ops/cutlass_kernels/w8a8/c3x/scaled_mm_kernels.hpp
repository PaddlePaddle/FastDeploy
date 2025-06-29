// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_kernels.hpp

#pragma once

#include "helper.h"

namespace fastdeploy {

void cutlass_scaled_mm_sm90_fp8(paddle::Tensor &out, paddle::Tensor const &a,
                                paddle::Tensor const &b,
                                paddle::Tensor const &a_scales,
                                paddle::Tensor const &b_scales,
                                paddle::optional<paddle::Tensor> const &bias);

void cutlass_scaled_mm_sm90_int8(paddle::Tensor &out, paddle::Tensor const &a,
                                 paddle::Tensor const &b,
                                 paddle::Tensor const &a_scales,
                                 paddle::Tensor const &b_scales,
                                 paddle::optional<paddle::Tensor> const &bias);

void cutlass_scaled_mm_azp_sm90_int8(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& b,
                                     paddle::Tensor const& a_scales,
                                     paddle::Tensor const& b_scales,
                                     paddle::Tensor const& azp_adj,
                                     paddle::optional<paddle::Tensor> const& azp,
                                     paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_sm100_fp8(paddle::Tensor &out, paddle::Tensor const &a,
                                 paddle::Tensor const &b,
                                 paddle::Tensor const &a_scales,
                                 paddle::Tensor const &b_scales,
                                 paddle::optional<paddle::Tensor> const &bias);

} // namespace fastdeploy
