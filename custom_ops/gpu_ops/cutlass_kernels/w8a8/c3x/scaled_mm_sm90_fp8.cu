// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu

// clang-format will break include orders
// clang-format off
#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm90_fp8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
// clang-format on

namespace fastdeploy {

void cutlass_scaled_mm_sm90_fp8(paddle::Tensor &out, paddle::Tensor const &a,
                                paddle::Tensor const &b,
                                paddle::Tensor const &a_scales,
                                paddle::Tensor const &b_scales,
                                paddle::optional<paddle::Tensor> const &bias) {
  PD_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  if (bias) {
    PD_CHECK(bias->dtype() == out.dtype(),
             "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm90_fp8_epilogue<c3x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm90_fp8_epilogue<c3x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}
} // namespace fastdeploy
