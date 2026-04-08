// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_azp_sm90_int8.cu

// clang-format will break include orders
// clang-format off
#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm90_int8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
// clang-format on

namespace fastdeploy {

void cutlass_scaled_mm_azp_sm90_int8(
    paddle::Tensor &out, paddle::Tensor const &a, paddle::Tensor const &b,
    paddle::Tensor const &a_scales, paddle::Tensor const &b_scales,
    paddle::Tensor const &azp_adj, paddle::optional<paddle::Tensor> const &azp,
    paddle::optional<paddle::Tensor> const &bias) {
  if (azp) {
    return cutlass_scaled_mm_sm90_int8_epilogue<
        c3x::ScaledEpilogueBiasAzpToken>(out, a, b, a_scales, b_scales, azp_adj,
                                         *azp, bias);
  } else {
    return cutlass_scaled_mm_sm90_int8_epilogue<c3x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

} // namespace fastdeploy
